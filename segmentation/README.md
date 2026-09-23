# Event segmentation for OpenCap motion capture data

Automated detection of movement events (labelled `S1`…`Sn`) across 18 standing, gait,
and balance tasks recorded with [OpenCap](https://www.opencap.ai/). Reads the `.trc`
marker files and `.mot` inverse-kinematics output an OpenCap session already produces,
and writes one events CSV per session plus diagnostic plots.

**Start with [`SOP_Run_Segmentation.md`](SOP_Run_Segmentation.md)** — a step-by-step
setup and run guide written for someone who has never used these scripts, Python, or
a terminal.

## Requirements

- Python 3.11 with OpenSim 4.6 (the SOP uses Miniconda; see Part 1)
- `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `requests`, `pyyaml`,
  `python-decouple`, `maskpass`
- A checkout of [`opencap-processing`](https://github.com/stanfordnmbl/opencap-processing)
  pinned to commit `72b5416bf6172fe3d9b42b01e1a02252362b20fc`

## Layout

The scripts locate `opencap-processing` automatically if it sits beside them:

```
OpenCapSegmentation/
├── opencap-processing/
└── segmentation-scripts/     <- the contents of this folder
```

Otherwise set `OPENCAP_PROCESSING_DIR` to point at the checkout.

## Running

```
python run_all.py "/path/to/SessionFolder"
```

Accepts several session folders at once. Results land in each session's `Outputs/`
folder: `Segmentation/events_<name>.csv` is the main deliverable, with marker and joint
angle plots alongside it for checking the result against the video.

## Relationship to OpenCap's own code

`segment_walking` and `segment_sts` are reused from `opencap-processing` essentially
verbatim, with a small number of deliberate substitutions. `verify_verbatim.py` re-checks
our copies against the originals and lists exactly what differs — run it after updating
`opencap-processing`. Every other detection threshold and the reasoning behind it is
documented in [`CHANGES_FROM_OPENCAP.md`](CHANGES_FROM_OPENCAP.md).
