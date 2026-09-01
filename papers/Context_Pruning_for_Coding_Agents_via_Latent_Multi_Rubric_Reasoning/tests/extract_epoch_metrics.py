#!/usr/bin/env python3
import argparse
import glob
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


AGGREGATE_TAGS = [
    "val/loss",
    "val/accuracy",
    "val/precision",
    "val/recall",
    "val/f1",
]

OBJECTIVE_TAGS = [
    "{name}_precision",
    "{name}_recall",
    "{name}_f1",
]


def latest_event_file(log_dir: str) -> str:
    paths = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))
    if not paths:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")
    return paths[-1]


def rounded(events):
    return [(item.step, round(item.value, 4)) for item in events]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract epoch metrics from TensorBoard logs.")
    parser.add_argument("log_dir", help="Directory containing events.out.tfevents.*")
    parser.add_argument(
        "--objectives",
        default="",
        help="Comma-separated objective names, e.g. semantic,dependency,context",
    )
    args = parser.parse_args()

    event_path = latest_event_file(args.log_dir)
    ea = EventAccumulator(event_path)
    ea.Reload()

    print(f"event_file: {event_path}")
    print("aggregate:")
    for tag in AGGREGATE_TAGS:
        if tag in ea.Tags().get("scalars", []):
            print(f"  {tag}: {rounded(ea.Scalars(tag))}")

    objectives = [name.strip() for name in args.objectives.split(",") if name.strip()]
    if objectives:
        print("objectives:")
        for objective in objectives:
            for pattern in OBJECTIVE_TAGS:
                tag = f"val/{pattern.format(name=objective)}"
                if tag in ea.Tags().get("scalars", []):
                    print(f"  {tag}: {rounded(ea.Scalars(tag))}")


if __name__ == "__main__":
    main()
