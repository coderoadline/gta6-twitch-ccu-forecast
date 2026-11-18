#!/usr/bin/env python3
"""Scale stress test for the launch-day forecast.

Each practice case is scaled so its hour-1 print sits at 1x, 5x, 10x and 30x
the largest comparable v1. The comparables and the momentum feature stay
fixed, so the median ratio curve should stay stable while the spread widens.
"""

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as model


MULTIPLIERS = (1, 5, 10, 30)
HOURS = (2, 6, 12, 18, 24)


def load_cases() -> list[tuple[str, dict, list[model.Comparable], float, float]]:
    cases = []
    for case_id in ("p00", "p01", "p02", "p03"):
        path = ROOT / "input" / f"{case_id}.in"
        case = json.loads(path.read_text(encoding="utf-8"))
        comparables = model.prepare_comparables(case)
        base_p = max(comp.v1 for comp in comparables)
        x = model._case_x(case)
        cases.append((case_id, case, comparables, base_p, x))
    return cases


def row_metrics(rows: list[list[float | int]], p: float) -> dict:
    q = [[float(value) for value in row] for row in rows]
    ratios = {hour: q[hour - 2][2] / p for hour in HOURS}
    widest = max(row[4] / row[0] for row in q)
    smallest_q05 = min(row[0] for row in q)
    jumps = [
        max(q[index + 1][2] / q[index][2], q[index][2] / q[index + 1][2])
        for index in range(len(q) - 1)
    ]
    return {
        "ratios": ratios,
        "widest": widest,
        "smallest_q05": smallest_q05,
        "max_jump": max(jumps),
        "q": q,
    }


def main() -> int:
    cases = load_cases()
    failures: list[str] = []
    max_jump = 0.0
    max_widest = 0.0
    extreme_ratios: dict[str, dict[int, float]] = {}

    for case_id, case, comparables, base_p, x in cases:
        for multiplier in MULTIPLIERS:
            p = base_p * multiplier
            case["known"][0]["viewers"] = p
            rows = model.forecast_hours(p, x, comparables, case_year=2026)
            metrics = row_metrics(rows, p)

            if multiplier == 5:
                extreme_ratios[case_id] = metrics["ratios"]
            if multiplier in (10, 30):
                for hour in HOURS:
                    base = extreme_ratios[case_id][hour]
                    relative = abs(metrics["ratios"][hour] / base - 1.0)
                    if relative > 0.03:
                        failures.append(
                            f"{case_id} hour {hour}: {multiplier}x ratio "
                            f"{metrics['ratios'][hour]:.4f} vs 5x {base:.4f} "
                            f"({relative:.1%})"
                        )

            for hour_index, row in enumerate(metrics["q"], start=2):
                for level_index, value in enumerate(row):
                    if not math.isfinite(value) or value <= 0:
                        failures.append(
                            f"{case_id} x{multiplier} hour {hour_index} q{level_index}: {value}"
                        )
                if any(row[index] > row[index + 1] for index in range(len(row) - 1)):
                    failures.append(f"{case_id} x{multiplier} hour {hour_index}: non-monotone")

            max_jump = max(max_jump, metrics["max_jump"])
            max_widest = max(max_widest, metrics["widest"])

            ratio_text = " ".join(
                f"h{hour}={metrics['ratios'][hour]:.4f}" for hour in HOURS
            )
            print(
                f"{case_id} x{multiplier:<2d} {ratio_text} "
                f"widest={metrics['widest']:.2f} "
                f"min_q05={metrics['smallest_q05']:.2f} "
                f"max_jump={metrics['max_jump']:.2f}"
            )

    if max_jump >= 2.0:
        failures.append(f"largest neighbouring median jump {max_jump:.3f} >= 2.0")
    if max_widest >= 12.0:
        failures.append(f"widest q95/q05 {max_widest:.3f} >= 12.0")

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
