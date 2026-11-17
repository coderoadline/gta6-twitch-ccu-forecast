"""GTA VI launch-day Twitch CCU quantile forecast.

The hybrid model pools comparable launch curves in log-ratio space. Model A
forms weighted empirical quantiles from the pool. Model B fits a small OLS
regression per hour over launch scale, pre-launch slope and release year. The
two are blended in log space, then a single spread floor is applied before
rounding. It emits five quantiles for each hour from 2 through 24.
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


def comparable_weight(comp: Comparable, p: float, x: float) -> float:
    scale_weight = math.exp(-0.5 * abs(math.log(comp.v1 / p)))
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
) -> list[float] | None:
    fit = _fit_regression(pool, hour)
    if fit is None:
        return None
    theta, residual_sd = fit
    median = (
        theta[0]
        + theta[1] * math.log(p)
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
    upper = model_a[3]
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


def forecast_hours(
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
    rows: list[list[float | int]] = []
    for hour in range(2, 25):
        model_a = _model_a_quantiles(p, pool, hour)
        if len(pool) < 6:
            blended = model_a
        else:
            model_b = _model_b_quantiles(p, x, case_year, pool, hour)
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
        blended = _widen_log_spread(blended, 1.1)
        rows.append(_finalize_quantiles(blended))
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
