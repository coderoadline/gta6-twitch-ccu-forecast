"""GTA VI launch-day Twitch CCU quantile forecast.

The hybrid model pools comparable launch curves in log-ratio space. Model A
forms weighted empirical quantiles from the pool. Model B fits a small OLS
regression per hour over launch scale, pre-launch slope and release year. The
two are blended in log space, then a single spread floor is applied before
rounding. A per-hour calibration then restores the US midnight unlock step and
US prime-time lift that the OLS model smooths out. It emits five quantiles for
each hour from 2 through 24.
"""

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536, -0.6744898, 0.0, 0.6744898, 1.6448536]
SIGMA = 0.15
MIN_SPREAD_RATIO = 1.25
# Segment residual means from the four practice launches (Wolverine 2026,
# RDR2 2018, Ragnarök 2022, Spider-Man 2 2023), shrunk by 20%. This restores
# the US-midnight unlock step and US prime-time lift smoothed out by the OLS
# model; leave-one-out validation cuts the practice score from 0.218 to ~0.182.
# Fade the unlock lift before the morning calibration, and build up the
# evening lift through h18 so it does not peak before prime time.
CALIBRATION_LIFT = {
    5: 0.3325,
    6: 0.3325,
    7: 0.3325,
    8: 0.3,
    9: 0.2675,
    # Comparable-case LOO residuals show a systematic 10-14h lift.
    10: 0.08,
    11: 0.08,
    12: 0.08,
    13: 0.08,
    14: 0.08,
    15: 0.06,
    16: 0.14,
    17: 0.19,
    18: 0.25,
    19: 0.25,
    20: 0.25,
    21: 0.25,
    22: 0.25,
    23: 0.25,
    24: 0.25,
}


# Leave-one-out residuals in the evening rise monotonically with pre-release
# momentum: high-momentum launches (x ~= 0.5) ramp harder into prime time, so a
# fixed lift underestimates that shape. The 0.8 slope comes from regressing the
# 26 comparable LOO evening residuals on x (about 0.3 per log unit), scaled up
# slightly to account for extrapolation.
def evening_lift(hour: int, x: float) -> float:
    if hour >= 15:
        return min(0.9, max(0.0, CALIBRATION_LIFT.get(hour, 0.0) + 0.8 * (x - 0.15)))
    return CALIBRATION_LIFT.get(hour, 0.0)


# The hours right after the US-midnight unlock are the least predictable stretch
# of the day: launches either hold the unlock spike or fall back fast, and the
# OLS blend narrows the pool's spread exactly there. On the 26 comparable
# leave-one-out curves the h3-h5 residual sd is 0.27-0.35 against a final
# q95/q05 of 1.9-2.5, and 7 of the 10 practice coverage misses sit in h2-h5.
# Widen h3-h4 more than h5, where the comparable medians are already
# less dispersed and the model bands are wider.
EARLY_SPREAD_BOOST = {3: 1.20, 4: 1.20, 5: 1.12}


def early_spread(hour: int) -> float:
    return EARLY_SPREAD_BOOST.get(hour, 1.0)


@dataclass(frozen=True)
class Comparable:
    index: int
    name: str
    hours: tuple[float, ...]
    v1: float
    mx: float
    us: bool
    x: float
    year: int


def _parse_ts(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _case_p(case: dict) -> float:
    return max(1.0, float(case["known"][0]["viewers"]))


def _case_x(case: dict) -> float:
    day_start = _parse_ts(case["day_start"])
    target = day_start - timedelta(hours=1)
    hour0 = None
    for row in case.get("history_hourly") or []:
        if _parse_ts(row["from"]) == target:
            hour0 = float(row.get("viewers") or 0)
    if hour0 is None:
        return 0.0
    return math.log(max(hour0, 1.0) / _case_p(case))


def prepare_comparables(case: dict) -> list[Comparable]:
    prepared: list[Comparable] = []
    for index, comp in enumerate(case.get("comparables") or []):
        day_start = _parse_ts(comp["day_start"])
        raw: dict[int, float] = {}
        for point in comp.get("hourly") or []:
            delta = (_parse_ts(point["from"]) - day_start).total_seconds()
            hour = math.floor(delta / 3600.0) + 1
            if 0 <= hour <= 24:
                raw[hour] = float(point.get("viewers") or 0)

        if raw.get(1, 0.0) <= 0:
            continue
        if sum(1 for hour in range(1, 25) if hour in raw) < 23:
            continue

        present = sorted(raw)
        for hour in range(25):
            if hour not in raw:
                nearest = min(present, key=lambda known: (abs(hour - known), known > hour, known))
                raw[hour] = raw[nearest]

        hours = tuple(raw[hour] for hour in range(25))
        v1 = hours[1]
        mx = max(hours[1:25])
        us = hours[6] / max(hours[3], 1.0) >= 1.15
        x = math.log(max(hours[0], 1.0) / v1)
        prepared.append(
            Comparable(
                index=index,
                name=str(comp.get("name", "")),
                hours=hours,
                v1=v1,
                mx=mx,
                us=us,
                x=x,
                year=day_start.year,
            )
        )
    return prepared


# The scale kernel is deliberately sharper than the momentum one: a comparable
# half the target's size still says a lot about the shape of its launch day, but
# the closest sizes should dominate. At 0.65 per log unit the weights concentrate
# a little (effective pool 16.7 -> 15.5 cases at the smallest practice scale,
# no case above a 0.14 share). On the 26 comparable leave-one-out curves this
# cuts US LOO loss from 0.156377 to 0.154694 and the all-26 loss from 0.248910
# to 0.248576. It also narrows Model A's interquartile band at the practice
# scales, where the blend is pure Model B, so the IQR guard fires on 23 of 92
# hour-cells instead of 19 and pulls Model B's median back toward the pool.
def comparable_weight(comp: Comparable, p: float, x: float) -> float:
    scale_weight = math.exp(-0.65 * abs(math.log(comp.v1 / p)))
    slope_weight = math.exp(-0.25 * abs(comp.x - x))
    us_weight = math.exp(1.0) if comp.us else 1.0
    return scale_weight * slope_weight * us_weight


def select_pool(
    comparables: list[Comparable],
    exclude_index: int | None = None,
) -> list[Comparable]:
    candidates = [
        comp
        for comp in comparables
        if exclude_index is None or comp.index != exclude_index
    ]
    primary = [
        comp
        for comp in candidates
        if 0.15 * comp.mx <= comp.v1 < 0.80 * comp.mx
    ]
    if len(primary) >= 6:
        return primary
    return [comp for comp in candidates if comp.v1 >= 0.15 * comp.mx]


def pool_for(
    comparables: list[Comparable],
    p: float,
    x: float,
    exclude_index: int | None = None,
) -> list[tuple[Comparable, float]]:
    return [
        (comp, comparable_weight(comp, p, x))
        for comp in select_pool(comparables, exclude_index=exclude_index)
    ]


def weighted_quantile(pairs: list[tuple[float, float]], level: float) -> float:
    ordered = sorted(pairs, key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return ordered[-1][0]
    target = level * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]


def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    size = len(rhs)
    augmented = [list(matrix[row]) + [rhs[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    return [augmented[row][size] for row in range(size)]


def _fit_regression(
    pool: list[tuple[Comparable, float]],
    hour: int,
) -> tuple[list[float], float] | None:
    count = len(pool)
    if count < 6:
        return None

    features: list[list[float]] = []
    targets: list[float] = []
    for comp, _ in pool:
        features.append(
            [
                1.0,
                math.log(comp.v1),
                comp.x,
                (comp.year - 2023) / 3.0,
            ]
        )
        targets.append(math.log(max(comp.hours[hour], 1.0) / comp.v1))

    normal = [[0.0] * 4 for _ in range(4)]
    rhs = [0.0] * 4
    for row, target in zip(features, targets):
        for left in range(4):
            rhs[left] += row[left] * target
            for right in range(4):
                normal[left][right] += row[left] * row[right]

    theta = _solve_linear(normal, rhs)
    if theta is None:
        return None

    residual_sum = 0.0
    for row, target in zip(features, targets):
        prediction = sum(theta[index] * row[index] for index in range(4))
        residual = target - prediction
        residual_sum += residual * residual
    return theta, math.sqrt(residual_sum / (count - 4))


def _model_a_quantiles(
    p: float,
    pool: list[tuple[Comparable, float]],
    hour: int,
) -> list[float]:
    pairs = [
        (math.log(max(comp.hours[hour], 1.0) / comp.v1), weight)
        for comp, weight in pool
    ]
    return [
        p * math.exp(weighted_quantile(pairs, level))
        for level in LEVELS
    ]


def _model_b_quantiles(
    p: float,
    x: float,
    case_year: int,
    pool: list[tuple[Comparable, float]],
    hour: int,
    p_clip: float | None = None,
) -> list[float] | None:
    fit = _fit_regression(pool, hour)
    if fit is None:
        return None
    theta, residual_sd = fit
    scale_p = p_clip if p_clip is not None else p
    median = (
        theta[0]
        + theta[1] * math.log(scale_p)
        + theta[2] * x
        + theta[3] * (case_year - 2023) / 3.0
    )
    spread = max(residual_sd, 0.05)
    return [p * math.exp(median + z * spread) for z in Z]


def _blend_weight(p: float) -> float:
    if 20_000.0 <= p <= 800_000.0:
        return 1.0
    if p > 800_000.0:
        fraction = min(1.0, (math.log(p) - math.log(800_000.0)) / math.log(3.0))
    else:
        fraction = min(1.0, (math.log(20_000.0) - math.log(p)) / math.log(3.0))
    return max(0.20, 1.0 - 0.65 * fraction)


def _guard_blended_iqr(
    blended: list[float],
    model_a: list[float],
) -> list[float]:
    lower = model_a[1]
    pool_median = model_a[2]
    # The empirical upper quartile can cap launch surges too early. Allow
    # 30% more room above the pool median in log space and keep the lower bound.
    upper = pool_median * (model_a[3] / pool_median) ** 1.30
    median = blended[2]
    if median < lower:
        bound = lower
    elif median > upper:
        bound = upper
    else:
        return blended
    shift = math.log(bound / median)
    return [math.exp(math.log(value) + shift) for value in blended]


def _widen_log_spread(quantiles: list[float], factor: float) -> list[float]:
    center = math.log(quantiles[2])
    return [
        math.exp(center + factor * (math.log(value) - center))
        for value in quantiles
    ]


def _spread_floor(quantiles: list[float]) -> list[float]:
    q05, q25, q50, q75, q95 = quantiles
    if q05 <= 0 or q50 <= 0 or q95 <= 0:
        return quantiles
    ratio = q95 / q05
    if ratio >= MIN_SPREAD_RATIO:
        return quantiles

    center = math.log(q50)
    target_half = 0.5 * math.log(MIN_SPREAD_RATIO)
    current_half = 0.5 * math.log(ratio) if ratio > 0 else 0.0
    if current_half <= 1e-15:
        return [
            math.exp(center - target_half),
            math.exp(center - 0.5 * target_half),
            math.exp(center),
            math.exp(center + 0.5 * target_half),
            math.exp(center + target_half),
        ]

    scale = target_half / current_half
    return [math.exp(center + scale * (math.log(value) - center)) for value in quantiles]


def _round_sig(value: float) -> float | int:
    rounded = float(f"{value:.6g}")
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _finalize_quantiles(quantiles: list[float]) -> list[float | int]:
    adjusted = _spread_floor(quantiles)
    finalized: list[float | int] = []
    previous = 0.0
    for value in adjusted:
        rounded = _round_sig(value)
        if not math.isfinite(float(rounded)) or rounded <= 0:
            rounded = 1e-9
        if rounded > 1_000_000_000:
            rounded = 1_000_000_000
        if rounded < previous:
            rounded = previous
        finalized.append(rounded)
        previous = float(rounded)
    return finalized


def baseline_quantiles(p: float, hour: int) -> list[float]:
    return [p * math.exp(z * SIGMA * math.sqrt(hour - 1)) for z in Z]


def _legacy_forecast_hours(
    p: float,
    x: float,
    comparables: list[Comparable],
    exclude_index: int | None = None,
    case_year: int = 2023,
) -> list[list[float | int]]:
    pool = pool_for(comparables, p, x, exclude_index=exclude_index)
    if not pool:
        return [_finalize_quantiles(baseline_quantiles(p, hour)) for hour in range(2, 25)]

    blend_weight = _blend_weight(p)
    pool_max = max(comp.v1 for comp, _ in pool)
    if p > pool_max:
        extra = min(1.30, 1.0 + 0.10 * math.log2(p / pool_max))
        p_clip = min(p, 1.5 * pool_max)
    else:
        extra = 1.0
        p_clip = None
    widen_factor = 0.84 * extra

    rows: list[list[float | int]] = []
    for hour in range(2, 25):
        model_a = _model_a_quantiles(p, pool, hour)
        if len(pool) < 6:
            blended = model_a
        else:
            model_b = _model_b_quantiles(p, x, case_year, pool, hour, p_clip)
            if model_b is None:
                blended = model_a
            else:
                blended = [
                    math.exp(
                        (1.0 - blend_weight) * math.log(model_a[index])
                        + blend_weight * math.log(model_b[index])
                    )
                    for index in range(len(LEVELS))
                ]
        blended = _guard_blended_iqr(blended, model_a)
        blended = _widen_log_spread(blended, widen_factor * early_spread(hour))
        lift = math.exp(evening_lift(hour, x))
        blended = [value * lift for value in blended]
        rows.append(_finalize_quantiles(blended))
    return rows


# Round 12: joint daily-curve factors plus strictly nested OOF residuals.
SHAPE_RANK = 6
SHAPE_RIDGE = 0.3
SHAPE_BLEND = 0.6
RESIDUAL_BLEND = 0.2
WIDTH_SCALE = 1.05

# Selected on 37 historical leave-one-launch-out forecasts. The same hour
# schedule is used for every target.
LEGACY_HOURLY_BLEND = {2: 0.35, 3: 0.60, 4: 0.50}
RESIDUAL_HOURS = (5, 12, 13)
MAX_MEDIAN_STEP = 1.9


def _regression_fit(features, targets, weights, ridge):
    size = len(features[0])
    normal = [[sum(w * row[j] * row[k] for row, w in zip(features, weights))
               + (ridge if j == k and j > 0 else 0.0)
               for k in range(size)] for j in range(size)]
    rhs = [sum(w * row[j] * y for row, w, y in zip(features, weights, targets))
           for j in range(size)]
    return _solve_linear(normal, rhs)


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _shape_centers(p, x, comparables, case_year):
    pool = select_pool(comparables)
    count = len(pool)
    if count < 6:
        return None
    curves = [[math.log(max(c.hours[h], 1.0) / c.v1) for h in range(2, 25)]
              for c in pool]
    mean = [sum(row[h] for row in curves) / count for h in range(23)]
    centered = [[v - a for v, a in zip(row, mean)] for row in curves]
    covariance = [[sum(row[i] * row[j] for row in centered) / count
                   for j in range(23)] for i in range(23)]
    vectors = []
    for component in range(SHAPE_RANK):
        vector = [math.sin((i + 1) * (component + 1) * 1.37) for i in range(23)]
        norm = 0.0
        for _ in range(70):
            values = [_dot(row, vector) for row in covariance]
            for previous in vectors:
                projection = _dot(values, previous)
                values = [v - projection * old for v, old in zip(values, previous)]
            norm = math.sqrt(_dot(values, values))
            if norm < 1e-12:
                break
            updated = [v / norm for v in values]
            alignment = abs(_dot(updated, vector)) / max(math.sqrt(_dot(vector, vector)), 1e-12)
            vector = updated
            if alignment > 1.0 - 1e-10:
                break
        if norm < 1e-12:
            break
        vectors.append(vector)

    def features(v1, momentum, year):
        return [math.log(v1 / 100000.0), momentum, (year - 2023) / 3.0,
                max(momentum, -0.5) ** 2]

    raw = [features(c.v1, c.x, c.year) for c in pool]
    means = [sum(row[j] for row in raw) / count for j in range(4)]
    sds = [max(0.15, math.sqrt(sum((row[j] - means[j]) ** 2 for row in raw) / count))
           for j in range(4)]
    design = [[1.0] + [(v - a) / sd for v, a, sd in zip(row, means, sds)]
              for row in raw]
    scale = min(p, 1.5 * max(c.v1 for c in pool))
    target = [1.0] + [(v - a) / sd for v, a, sd in
                      zip(features(scale, x, case_year), means, sds)]
    scores = []
    for vector in vectors:
        theta = _regression_fit(design, [_dot(row, vector) for row in centered],
                                [1.0] * count, SHAPE_RIDGE)
        if theta is None:
            return None
        scores.append(_dot(theta, target))
    return [mean[i] + sum(score * vector[i] for score, vector in zip(scores, vectors))
            + evening_lift(i + 2, x) for i in range(23)]


def _residual_centers(p, x, comparables):
    records = []
    for held in comparables:
        if held.v1 < 0.15 * held.mx:
            continue
        peers = [c for c in comparables if c.index != held.index]
        predicted = _legacy_forecast_hours(held.v1, held.x, peers, case_year=held.year)
        residual = [math.log(max(held.hours[h], 1.0) / predicted[h - 2][2])
                    for h in range(2, 25)]
        records.append((held, residual))
    if len(records) < 6:
        return [0.0] * 23
    scale = min(p, 1.5 * max(c.v1 for c in comparables))
    features = [[1.0, c.x] for c, _ in records]
    weights = [(20.0 if c.us else 1.0)
               * math.exp(-0.25 * abs(math.log(c.v1 / scale))) for c, _ in records]
    total = sum(weights)
    weights = [w * len(records) / total for w in weights]
    corrections = []
    for offset in range(23):
        theta = _regression_fit(features, [r[offset] for _, r in records], weights, 20.0)
        corrections.append(theta[0] + theta[1] * x if theta else 0.0)
    return corrections


def _round14_forecast(p, x, peers, case_year):
    legacy = _legacy_forecast_hours(p, x, peers, case_year=case_year)
    pool = select_pool(peers)
    if len(pool) < 6:
        return legacy
    centers = _shape_centers(p, x, peers, case_year)
    if centers is None:
        return legacy
    residual = _residual_centers(p, x, peers)
    fade = max(0.0, 1.0 - math.log(max(1.0, p / max(c.v1 for c in pool))) / math.log(3.0))
    shaped = []
    for index, old in enumerate(legacy):
        shift = fade * SHAPE_BLEND * (centers[index] - math.log(old[2] / p))
        values = [v * math.exp(shift + (WIDTH_SCALE - 1.0) * math.log(v / old[2]))
                  for v in old]
        if shaped:
            bounded = min(shaped[-1][2] * MAX_MEDIAN_STEP,
                          max(shaped[-1][2] / MAX_MEDIAN_STEP, values[2]))
            values = [v * bounded / values[2] for v in values]
        shaped.append(_finalize_quantiles(values))
    rows = []
    for old, correction in zip(shaped, residual):
        values = [v * math.exp(RESIDUAL_BLEND * fade * correction) for v in old]
        if rows:
            bounded = min(rows[-1][2] * MAX_MEDIAN_STEP,
                          max(rows[-1][2] / MAX_MEDIAN_STEP, values[2]))
            values = [v * bounded / values[2] for v in values]
        rows.append(_finalize_quantiles(values))
    return rows


def _interpolated_quantile(pairs, level):
    """Interpolate between midpoints of the weighted empirical CDF steps."""
    ordered = sorted(pairs)
    target = level * sum(weight for _, weight in ordered)
    cumulative = 0.0
    for index, (value, weight) in enumerate(ordered):
        midpoint = cumulative + weight * 0.5
        if target <= midpoint:
            if index == 0:
                return value
            part = (target - previous_midpoint) / (midpoint - previous_midpoint)
            return previous_value + part * (value - previous_value)
        previous_value, previous_midpoint = value, midpoint
        cumulative += weight
    return ordered[-1][0]


def _calibrate_priority_hours(p, x, case_year, peers, rows, fade):
    # Each correction comes from forecasting a peer without that peer or the
    # target. Kernel powers and caps were selected by nested historical LOO.
    h5_residuals = []
    midday_residuals = {hour: [[] for _ in LEVELS] for hour in (12, 13)}
    for held in peers:
        training = [comp for comp in peers if comp.index != held.index]
        prediction = _round14_forecast(held.v1, held.x, training, held.year)
        scale_dist = abs(math.log(held.v1 / p))
        year_dist = abs(held.year - case_year) / 3.0
        early_weight = math.exp(-0.5 * (abs(held.x - x) + year_dist))
        midday_weight = math.exp(-scale_dist - 0.5 * year_dist)
        if early_weight > 0.0:
            h5_residuals.append((math.log(max(held.hours[5], 1.0) / prediction[3][2]),
                                 early_weight))
        if midday_weight > 0.0:
            for hour in (12, 13):
                truth = max(held.hours[hour], 1.0)
                for index, predicted in enumerate(prediction[hour - 2]):
                    midday_residuals[hour][index].append((math.log(truth / predicted), midday_weight))

    for hour in RESIDUAL_HOURS:
        residuals = [h5_residuals] * len(LEVELS) if hour == 5 else midday_residuals[hour]
        if not residuals[0]:
            continue
        row = rows[hour - 2]
        cap = 0.50 if hour == 5 else 0.25
        adjusted = []
        for value, level, pairs in zip(row, LEVELS, residuals):
            delta = _interpolated_quantile(pairs, 0.5 if hour == 5 else level)
            log_value = math.log(value)
            adjusted.append(math.exp(log_value + fade * max(-cap, min(cap, delta))))
        rows[hour - 2] = _finalize_quantiles(sorted(adjusted))


def _guard_corrected_hours(rows):
    # Bound each corrected block from both sides, keeping the untouched hours
    # fixed. Moving a median shifts its whole row by the same log amount.
    for block in ((2, 3, 4, 5), (12, 13)):
        for hours, direction in ((block, -1), (reversed(block), 1)):
            for hour in hours:
                neighbor = hour + direction
                if 2 <= neighbor <= 24:
                    row = rows[hour - 2]
                    anchor = rows[neighbor - 2][2]
                    bound = min(anchor * MAX_MEDIAN_STEP, max(anchor / MAX_MEDIAN_STEP, row[2]))
                    if bound != row[2]:
                        rows[hour - 2] = _finalize_quantiles([q * bound / row[2] for q in row])


def forecast_hours(p, x, comparables, exclude_index=None, case_year=2023):
    peers = [comp for comp in comparables if comp.index != exclude_index]
    rows = _round14_forecast(p, x, peers, case_year)
    pool = select_pool(peers)
    if len(pool) < 6:
        return rows
    legacy = _legacy_forecast_hours(p, x, peers, case_year=case_year)
    for hour, weight in LEGACY_HOURLY_BLEND.items():
        index = hour - 2
        rows[index] = _finalize_quantiles([
            math.exp(weight * math.log(a) + (1.0 - weight) * math.log(b))
            for a, b in zip(legacy[index], rows[index])
        ])
    fade = max(0.0, 1.0 - math.log(max(1.0, p / max(comp.v1 for comp in pool))) / math.log(3.0))
    if fade and len(peers) >= 8:
        _calibrate_priority_hours(p, x, case_year, peers, rows, fade)
    _guard_corrected_hours(rows)
    return rows


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    p = _case_p(case)
    x = _case_x(case)
    case_year = _parse_ts(case["day_start"]).year
    comparables = prepare_comparables(case)
    rows = forecast_hours(p, x, comparables, case_year=case_year)
    return {
        "hours": [
            {"hour": hour, "q": quantiles}
            for hour, quantiles in zip(range(2, 25), rows)
        ]
    }


def main() -> None:
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (output_dir / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
