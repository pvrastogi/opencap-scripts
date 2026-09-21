# Event segmentation scripts for OpenCap sessions

These scripts find and label the events inside each OpenCap task — heel strikes,
toe-offs, sit-to-stand phases, reach onsets and returns, stance touchdowns — and write
them out as one row per event, together with marker and joint-angle plots.

**Start with the SOP. It covers installing
OpenSim, downloading the pinned `opencap-processing` checkout these scripts depend on,
naming the trials correctly, and reading the output. Running anything here without that
setup will fail at the first import.

## Quick reference

You only ever run one script:

```
python run_all.py "/path/to/patient/session/folder"
```

It runs five steps per session — manifest, spike check, segmentation, marker plots,
coordinate plots — and writes everything into that session's own `Outputs/` folder.

| File | Role |
| --- | --- |
| `run_all.py` | The only script you run. Calls the five steps below. |
| `manifest.py` | What is on disk vs what we segment. |
| `flag_spikes.py` | Trials with physically impossible marker jumps. |
| `run_segmentation.py` | Runs every task module; writes the events CSV, run log and overlays. |
| `plot_opencap_trc_markers.py` | Marker position charts. |
| `plot_opencap_all_coords.py` | Joint angle charts. |
| `seg_common.py` | Shared plumbing: file loading, calibration trim, turn detection, order and interval checks. |
| `seg_walk_core.py`, `seg_sts_core.py` | Copies of OpenCap's own gait and sit-to-stand routines, with our documented changes. |
| `seg_*.py` | One module per task family; each file's docstring defines its events. |
| `plot_events_overlay.py` | Draws events onto the charts. |
| `verify_verbatim.py` | Maintainer tool: re-checks our copies of OpenCap code against the originals. |
| `CHANGES_FROM_OPENCAP.md` | Every deliberate difference from stock OpenCap, with the measurements behind each one. |

## Dependencies

Python 3.11 with `opensim` 4.6 (conda, `opensim-org` channel — not pip-installable),
plus `numpy scipy pandas matplotlib seaborn requests pyyaml python-decouple maskpass`,
and a checkout of `stanfordnmbl/opencap-processing` pinned to commit `72b5416`. The SOP
walks through all of it.
