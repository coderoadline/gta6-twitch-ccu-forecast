"""GTA VI launch-day Twitch CCU quantile forecast.

The model pools comparable launch curves in log-ratio space, weighting each
comparable by how closely its hour-1 scale and pre-launch slope match the case.
It emits five quantiles for each hour from 2 through 24.
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
            )
        )
    return prepared


def comparable_weight(comp: Comparable, p: float, x: float) -> float:
    scale_weight = math.exp(-0.7 * abs(math.log(comp.v1 / p)))
    slope_weight = math.exp(-0.8 * abs(comp.x - x))
    us_weight = math.exp(0.8) if comp.us else 1.0
    return scale_weight * slope_weight * us_weight


def pool_for(
    comparables: list[Comparable],
    p: float,
    x: float,
    exclude_index: int | None = None,
) -> list[tuple[Comparable, float]]:
    pool: list[tuple[Comparable, float]] = []
    for comp in comparables:
        if exclude_index is not None and comp.index == exclude_index:
            continue
        if comp.v1 < 0.15 * comp.mx:
            continue
        pool.append((comp, comparable_weight(comp, p, x)))
    return pool


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
) -> list[list[float | int]]:
    pool = pool_for(comparables, p, x, exclude_index=exclude_index)
    if not pool:
        return [_finalize_quantiles(baseline_quantiles(p, hour)) for hour in range(2, 25)]

    rows: list[list[float | int]] = []
    for hour in range(2, 25):
        pairs = [
            (math.log(max(comp.hours[hour], 1.0) / comp.v1), weight)
            for comp, weight in pool
        ]
        raw = [
            p * math.exp(weighted_quantile(pairs, level))
            for level in LEVELS
        ]
        rows.append(_finalize_quantiles(raw))
    return rows


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    p = _case_p(case)
    x = _case_x(case)
    comparables = prepare_comparables(case)
    rows = forecast_hours(p, x, comparables)
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
