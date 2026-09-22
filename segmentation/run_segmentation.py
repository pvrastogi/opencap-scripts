import csv
import json
import os
import sys

import seg_common as sc
import seg_standing
import seg_rtt
import seg_tandem
import seg_sls
import seg_5tsts
import seg_march
import seg_gait
import seg_gait_bckw
import seg_gait_pivot
import seg_frt
import seg_ybalance
import seg_arms
import seg_tug
import plot_events_overlay as overlay

# Ordered as the segmentation markers document orders its tables: steady
# tasks, then cyclic, then discrete.
MODULES = [
    # steady
    ("standing", seg_standing),
    ("rise to toes", seg_rtt),
    ("tandem stance", seg_tandem),
    ("single-leg stance", seg_sls),
    # cyclic
    ("sit-to-stand", seg_5tsts),
    ("march", seg_march),
    ("walking", seg_gait),
    ("backward walking", seg_gait_bckw),
    ("pivot", seg_gait_pivot),
    # discrete
    ("functional reach", seg_frt),
    ("y-balance", seg_ybalance),
    ("arm raises", seg_arms),
    ("timed up and go", seg_tug),
]


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "PR"
    sd = sc.session_dir(label)
    # label may be a path; outputs are named after it
    out = sc.short_label(label)
    out_dir = os.path.join(sd, "Outputs", "Segmentation")
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== {out}")
    print(f"Session: {sd}\n")

    all_events, all_meta = [], []
    for group, module in MODULES:
        print(f"[{group}]")
        events, meta = module.run(label)
        all_events += events
        all_meta += meta
        print()

    all_events = sc.sort_events(all_events)

    csv_path = os.path.join(out_dir, f"events_{out}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sc.EVENT_FIELDS)
        w.writeheader()
        for e in all_events:
            w.writerow(e)

    log_path = os.path.join(out_dir, f"run_log_{out}.txt")
    with open(log_path, "w") as fh:
        fh.write(f"session: {sd}\n\n")
        for m in all_meta:
            fh.write(json.dumps(m, indent=2, default=str) + "\n")

    print("--- events per task")
    by_task = {}
    for e in all_events:
        by_task.setdefault(e["task"], []).append(e)
    for task in sorted(by_task):
        print(f"  {task:16s} {len(by_task[task]):3d}")
    failures = [m for m in all_meta if "error" in m]
    print(f"\n{len(all_events)} events across {len(by_task)} tasks, "
          f"{len(failures)} task failures")
    for m in failures:
        print(f"  FAILED {m['task']}: {m['error']}")

    print("\n--- consecutiveness check")
    violations = sc.check_order(all_events)
    if not violations:
        print("  OK: in every task, no S(N+1) lands before its S(N)")
    else:
        for v in violations:
            print(f"  {v['task']:16s} [{v['kind']}] {v['detail']}")
    with open(log_path, "a") as fh:
        fh.write("\nconsecutiveness violations: "
                 + (json.dumps(violations, indent=2) if violations else "none")
                 + "\n")

    print("\n--- stride interval outliers")
    outliers = sc.flag_interval_outliers(all_events)
    if not outliers:
        print(f"  OK: no gait interval exceeds {sc.INTERVAL_RATIO}x the "
              f"median for its event pair")
    else:
        for o in outliers:
            anchor = f" anchor {o['anchor_leg']}" if o["anchor_leg"] else ""
            print(f"  {o['task']:16s} cycle {o['cycle']}{anchor}: "
                  f"{o['from_event']}->{o['to_event']} is {o['interval_s']}s "
                  f"at {o['at_s']}s, {o['ratio']}x the {o['median_s']}s "
                  f"median -- check the video")
    with open(log_path, "a") as fh:
        fh.write("\nstride interval outliers: "
                 + (json.dumps(outliers, indent=2) if outliers else "none")
                 + "\n")

    print(f"\nwrote {csv_path}")
    print(f"wrote {log_path}\n")

    print("--- overlays")
    overlay.overlay_all(label, by_task)


if __name__ == "__main__":
    main()
