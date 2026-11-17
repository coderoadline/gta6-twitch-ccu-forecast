#!/usr/bin/env python3
"""Leave-one-out validation for the comparable-pool forecast."""

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as model


def pinball_loss(y: float, q: float, tau: float) -> float:
    d = math.log(max(float(y), 1.0)) - math.log(float(q))
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def load_case() -> dict:
    input_dir = ROOT / "input"
    path = input_dir / "p00.in"
    if not path.exists():
        candidates = sorted(input_dir.glob("*.in"))
        if not candidates:
            raise SystemExit("no input/*.in case found")
        path = candidates[0]
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    case = load_case()
    comparables = model.prepare_comparables(case)
    launched = [comp for comp in comparables if comp.v1 >= 0.15 * comp.mx]
    model_total = 0.0
    baseline_total = 0.0
    items = 0

    for held_out in launched:
        pool = [comp for comp in comparables if comp.index != held_out.index]
        forecast = model.forecast_hours(held_out.v1, held_out.x, pool)

        for hour, quantiles in zip(range(2, 25), forecast):
            y = held_out.hours[hour]
            for tau, q in zip(model.LEVELS, quantiles):
                model_total += pinball_loss(y, q, tau)
            for tau, q in zip(model.LEVELS, model.baseline_quantiles(held_out.v1, hour)):
                baseline_total += pinball_loss(y, q, tau)
            items += len(model.LEVELS)

    print(f"kept comparables: {len(comparables)}")
    print(f"launched held-out comparables: {len(launched)}")
    print(f"LOO model pinball loss: {model_total / items:.6f}")
    print(f"LOO baseline pinball loss: {baseline_total / items:.6f}")


if __name__ == "__main__":
    main()
