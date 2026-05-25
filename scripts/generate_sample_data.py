"""Generate a synthetic work-order Excel file for testing the engine.

Produces ~500 rows with several realistic failure categories, deliberately
introduces some mismatched descriptions (e.g. an HVAC chiller order whose
description talks about replacing a light bulb) so you can confirm the
confidence-scoring step actually catches them.

Usage:
    python scripts/generate_sample_data.py --out data/sample_work_orders.xlsx
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

CATEGORIES = {
    "HVAC": {
        "equipment": ["Chiller", "AHU", "RTU", "Boiler", "Cooling Tower", "VAV Box"],
        "trade": "Mechanical",
        "descriptions": [
            "Chiller compressor short cycling, alarm code 23",
            "AHU supply fan belt slipping, replace belt",
            "Cooling tower fill clogged with scale buildup",
            "Boiler ignition failure on startup",
            "Filter pressure drop high, swap pleated filters",
            "VAV box damper actuator not responding",
            "Refrigerant leak detected at condenser line",
            "Hot water pump bearing noise, lubricate",
            "Thermostat reading 5 deg off, recalibrate sensor",
            "Condensate drain pan overflowing",
        ],
    },
    "Electrical": {
        "equipment": ["Breaker Panel", "Transformer", "UPS", "Switchgear", "Outlet"],
        "trade": "Electrical",
        "descriptions": [
            "Breaker tripping under load on panel B3",
            "UPS battery showing end of life warning",
            "Outlet sparking when plugging in vacuum",
            "Transformer humming louder than normal",
            "Emergency lighting battery test failed",
            "GFCI outlet will not reset",
            "Panel feed conductor showing heat discoloration",
            "Switchgear contactor stuck closed",
        ],
    },
    "Plumbing": {
        "equipment": ["Toilet", "Sink", "Water Heater", "Sump Pump", "Backflow Preventer"],
        "trade": "Plumbing",
        "descriptions": [
            "Toilet running constantly, replace flapper",
            "Hot water heater leaking from T&P valve",
            "Sump pump float stuck, basement flooding risk",
            "Sink faucet drip 1 drop per second",
            "Backflow preventer annual test due",
            "Slow drain in mens room sink",
            "Water heater pilot light keeps going out",
            "Pipe sweating heavily in mechanical room",
        ],
    },
    "Lighting": {
        "equipment": ["LED Fixture", "Ballast", "Exit Sign", "Emergency Light"],
        "trade": "Electrical",
        "descriptions": [
            "LED fixture flickering in hallway",
            "Replace burnt out tube in office 204",
            "Exit sign LED dim, swap module",
            "Emergency light not illuminating during test",
            "Ballast humming, replace",
            "Photocell not turning on parking lot lights at dusk",
        ],
    },
    "Fire/Life Safety": {
        "equipment": ["Smoke Detector", "Sprinkler", "Fire Pump", "FACP"],
        "trade": "Fire Protection",
        "descriptions": [
            "Smoke detector chirping low battery",
            "Sprinkler head painted over, replace",
            "Fire pump weekly churn test failed",
            "FACP showing trouble on zone 4",
            "Pull station cover broken",
        ],
    },
    "Elevator": {
        "equipment": ["Passenger Elevator", "Freight Elevator"],
        "trade": "Elevator",
        "descriptions": [
            "Elevator door re-opening repeatedly",
            "Cab leveling off by 2 inches at lobby",
            "Phone in cab not connecting to monitoring",
            "Buttons unresponsive on second floor",
        ],
    },
    "Roofing": {
        "equipment": ["Roof Membrane", "Roof Drain"],
        "trade": "General",
        "descriptions": [
            "Ponding water on roof near drain 3",
            "Membrane seam lifting near HVAC curb",
            "Roof drain clogged with leaves",
        ],
    },
}

LOCATIONS = [
    "Floor 1", "Floor 2", "Floor 3", "Mechanical Room",
    "Roof", "Basement", "Parking Garage", "Lobby",
]
PRIORITIES = ["Low", "Medium", "High", "Urgent"]

UNRELATED_DESCRIPTIONS = [
    "replaced light bulb in office",
    "swept the floor",
    "looked at it, seems fine",
    "n/a",
    "see attached",
    "done",
    "called vendor",
    "moved a chair into the meeting room",
    "ordered parts, will return",
]


def generate(n: int, mismatch_rate: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    cats = list(CATEGORIES.keys())
    weights = [0.30, 0.18, 0.15, 0.14, 0.10, 0.08, 0.05]

    for i in range(n):
        cat = rng.choices(cats, weights=weights, k=1)[0]
        info = CATEGORIES[cat]
        equip = rng.choice(info["equipment"])
        desc = rng.choice(info["descriptions"])

        is_mismatch = rng.random() < mismatch_rate
        if is_mismatch:
            desc = rng.choice(UNRELATED_DESCRIPTIONS)

        rows.append(
            {
                "WorkOrderID": f"WO-{10000 + i}",
                "AssetCategory": cat,
                "EquipmentType": equip,
                "Location": rng.choice(LOCATIONS),
                "Trade": info["trade"],
                "Priority": rng.choices(PRIORITIES, weights=[0.4, 0.35, 0.2, 0.05])[0],
                "Description": desc,
                "ResolutionNotes": "" if rng.random() < 0.3 else "Repaired and tested",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/sample_work_orders.xlsx")
    p.add_argument("--n", type=int, default=500, help="Number of rows to generate.")
    p.add_argument(
        "--mismatch-rate",
        type=float,
        default=0.08,
        help="Fraction of rows whose description is intentionally unrelated.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = generate(args.n, args.mismatch_rate, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False, engine="openpyxl")
    print(
        f"Wrote {len(df)} rows to {out} "
        f"(~{int(args.mismatch_rate * 100)}% intentionally mismatched descriptions)"
    )


if __name__ == "__main__":
    main()
