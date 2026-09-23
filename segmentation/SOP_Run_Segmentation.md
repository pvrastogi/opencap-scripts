# SOP: Running Event Segmentation on OpenCap Patient Data

**Purpose:** This document walks you through, step by step, how to run our automated
event-segmentation scripts on any patient's OpenCap data. It assumes you have never
written or run code before. Follow every step in order the first time. After your
computer is set up (Part 1), you only need Parts 2–5 for each new patient.

**Who this is for:** Anyone on the team who needs to produce segmentation event tables
and plots from an OpenCap session, with no coding background required.

**Read the QC SOP first.** `SOP_Run_QC.md` checks whether a session's data is *trustworthy*.
This SOP labels *what happened when* inside each task. Run QC first; if a trial fails QC
badly, segmenting it will produce confident-looking nonsense.

---

## Overview: what this actually does

Every OpenCap session produces marker positions (`.trc`) and joint angles (`.mot`) for
each task the patient performed. **Segmentation** means finding the exact instant each
meaningful event happened inside a task — the moment a heel strikes the floor, the moment
a patient finishes standing up, the moment a reach comes back to neutral — and writing
them out as a table.

Each event gets a label `S1`, `S2`, `S3` … in the order the segmentation markers document
defines them for that task. For repeating tasks like walking, `S1`–`S4` repeat once per
gait cycle, with a cycle number.

You will run **one script**, `run_all.py`, which does five things in order:

1. **Manifest** — lists what tasks are on disk, which we segment, and which we would
   otherwise silently ignore.
2. **Spike check** — flags trials where a marker physically teleports between frames.
3. **Segmentation** — produces the event table, a detailed run log, and annotated plots.
4. **Marker plots** — one chart per trial of every marker's position over time.
5. **Coordinate (angle) plots** — one chart per trial of every joint angle over time.

Everything is written inside the patient's own folder, under `Outputs/`. Nothing is
written anywhere else.

> **Why this needs more setup than the QC SOP.** These scripts read OpenSim models to
> get joint angles and body orientations. OpenSim **cannot be installed with `pip`** — it
> needs a different installer called conda. That is the only real difference, and Part 1
> handles it.

---

## Part 1 — One-time computer setup

*Skip to Part 2 if this computer has already been set up.*

### 1a. Install Visual Studio Code (VS Code)

VS Code is the program used to open the code and run it.

- Go to `https://code.visualstudio.com/` and click **Download**.
- **Mac:** a `.zip` downloads. Double-click to unzip, then drag the **Visual Studio Code**
  app into your **Applications** folder.
- **Windows:** a `.exe` installer downloads. Double-click it, accept the license, click
  **Next** through the default options, then **Install** → **Finish**.
- Open VS Code once to confirm it launches. You can close it after confirming.

### 1b. Install Miniconda

Conda is an installer that can install OpenSim. Plain Python cannot.

- Go to `https://www.anaconda.com/download/success` and scroll to **Miniconda Installers**.
- **Mac:** download the **pkg** installer. Pick **Apple Silicon** if your Mac is M1/M2/M3/M4,
  or **Intel** otherwise. (Apple menu → About This Mac tells you which.) Open the `.pkg`
  and click **Continue → Continue → Agree → Install**, entering your Mac password if asked.
- **Windows:** download the **Miniconda3 Windows 64-bit** `.exe`. Run it and click through
  with the default options.
- **Close VS Code completely and reopen it.** Conda is not available until you do.

**Verify the install:**

1. Open VS Code.
2. Open the terminal: **Terminal → New Terminal** (a text panel appears at the bottom).
3. Type `conda --version` and press Enter.
4. You should see something like `conda 25.1.1`. If you see "command not found", close
   and reopen VS Code and try again. If it still fails, re-run the installer.

### 1c. Create the environment

An "environment" is a private, self-contained set of software so these scripts cannot
break anything else on your computer. Create it once.

First, move to your home folder. If the terminal is sitting in some other project folder
— especially one under iCloud Drive, Desktop, or Documents — macOS can block conda from
reading the current directory, and it crashes before it does anything. In the VS Code
terminal, paste this and press Enter:

```
cd ~
```

Now paste this **one line** and press Enter:

```
conda create -y -n opensim_env -c opensim-org python=3.11 opensim=4.6
```

This takes several minutes and prints a lot of text. It is done when you get a new blank
line back.

Now **activate** it and install the remaining packages. Paste these two lines one at a
time:

```
conda activate opensim_env
```

```
pip install numpy scipy pandas matplotlib seaborn requests pyyaml python-decouple maskpass
```

> **What "activate" means.** Your terminal prompt will change to start with
> `(opensim_env)`. That tells you the environment is switched on. **You must run
> `conda activate opensim_env` every time you open a new terminal** before running these
> scripts. If you forget, you will get a `ModuleNotFoundError`.

**Verify:** paste this one line and press Enter.

```
python -c "import opensim, numpy, scipy, pandas, matplotlib; print('setup OK', opensim.__version__)"
```

You should see `setup OK 4.6`. If you see an error, redo step 1c.

### 1d. Make a working folder

Create one folder to hold the code. It can live anywhere you like — your Desktop is fine.
Call it `OpenCapSegmentation`. Everything in steps 1e and 1f goes inside it.

### 1e. Download opencap-processing

Our scripts reuse the OpenCap team's own gait and sit-to-stand analysis code. You need
their code alongside ours.

1. Open this exact link in your browser. **Do not** use their main download page — this
   link pins the specific version our scripts were written and tested against:

   `https://github.com/stanfordnmbl/opencap-processing/archive/72b5416bf6172fe3d9b42b01e1a02252362b20fc.zip`

2. Unzip it. It unzips to a folder named
   `opencap-processing-72b5416bf6172fe3d9b42b01e1a02252362b20fc`.
3. **Rename that folder to exactly `opencap-processing`** (delete everything from the
   hyphen after "processing" onward).
4. Move it inside your `OpenCapSegmentation` folder.

### 1f. Download the segmentation scripts

1. Go to the repo: `https://github.com/pvrastogi/opencap-scripts`
2. Click the green **Code** button.
3. Click **Download ZIP**.
4. Unzip it. It unzips into a folder named `opencap-scripts-main`.
5. Inside it, open the `segmentation` folder. Copy **all** of its files.
6. Create a folder called `segmentation-scripts` inside `OpenCapSegmentation` and paste
   the files there.

> **Important:** copy the *files* out of the `segmentation` folder. Do not paste the
> `opencap-scripts-main` folder itself, and do not rename any script. Keep the unzipped
> copy so you can reuse it for the next computer, and re-download whenever the scripts
> are updated. The other files in `opencap-scripts-main` (the QC scripts, the
> shoulder-reprocessing scripts) are for other SOPs and can be ignored here.

### 1g. Check the layout

Your folder must look exactly like this — the two folders **side by side**:

```
OpenCapSegmentation/
├── opencap-processing/          (from step 1e)
│   ├── ActivityAnalyses/
│   ├── utilsKinematics.py
│   └── ... many other files
└── segmentation-scripts/        (from step 1f)
    ├── run_all.py
    ├── run_segmentation.py
    ├── seg_common.py
    └── ... about 25 files in total
```

If `opencap-processing` ends up *inside* `segmentation-scripts`, that also works. What
does **not** work is putting them in unrelated places.

### 1h. Verify the whole setup

In the VS Code terminal, run these two lines. Replace the path in the first line with the
path to your own `segmentation-scripts` folder.

```
cd "/Users/yourname/Desktop/OpenCapSegmentation/segmentation-scripts"
```

```
python -c "import seg_common as sc, os; print('scripts OK'); print('opencap-processing found:', os.path.isdir(os.path.join(sc.OPENCAP_DIR,'ActivityAnalyses')), '->', sc.OPENCAP_DIR)"
```

You want to see `scripts OK` and `opencap-processing found: True`.

If it says `False`, the two folders are not arranged as in 1g. Either fix the layout, or
tell the scripts where the checkout is by running this once per terminal session,
substituting your own path:

```
export OPENCAP_PROCESSING_DIR="/Users/yourname/Desktop/OpenCapSegmentation/opencap-processing"
```

(On Windows, use `set OPENCAP_PROCESSING_DIR=C:\path\to\opencap-processing` instead.)

**Part 1 is now complete for this computer and does not need to be repeated.**

---

## Part 2 — Prepare a new patient's data

*Repeat this part for every patient.*

### 2a. Download the session from OpenCap

1. Go to `opencap.ai` and log in.
2. Find the patient's session. In the **Actions** column, click the left-most button (the
   play/triangle icon).
3. At the bottom of the task list, click the up-arrow icon.
4. Click **Download data**. A `.zip` downloads.

### 2b. Unzip and name the folder

- **Mac:** double-click the `.zip`. It unzips into `OpenCapData_<long string>`.
- **Windows:** right-click → **Extract All…** → choose a location → **Extract**.
- Move the folder somewhere you can find it and **rename it to something you will
  recognise** — for example the patient's study ID.

Confirm the folder contains `MarkerData`, `OpenSimData`, and `Videos`, plus
`sessionMetadata.yaml`. If any are missing, the download failed; repeat 2a.

> **This renamed folder is "the patient folder". Every command below points at it.**

### 2c. Check the trial names — do not skip this

The scripts find each task **by its file name**. A trial named `walking` instead of
`gait` will be silently ignored. This is the single most common cause of missing results.

Open the patient folder → `MarkerData`. You will see one `.trc` file per task. Rename them
so they match this list exactly (lower-case, hyphens not spaces, no numbers on the end):

| Task file name | What it is |
| --- | --- |
| `standing-ec` | Standing, eyes closed |
| `rtt` | Rise to toes |
| `semi-tandem` | Semi-tandem stance |
| `tandem` | Tandem stance |
| `sls-ec` | Single-leg stance, eyes closed |
| `5tsts` | Five times sit-to-stand |
| `march` | Marching in place |
| `gait` | Walking |
| `gait-start` | Gait initiation |
| `gait-head-turn` | Walking with head turns |
| `gait-tandem` | Tandem walking |
| `gait-bckw` | Walking backwards |
| `gait-bckw-3s` | Walking backwards, 3-second version |
| `gait-pivot` | Walking with a pivot turn |
| `frt` | Functional reach test |
| `y-balance` | Y-balance |
| `arms` | Arm raises in four planes |
| `tug` | Timed up and go |

**Rename the matching `.mot` file too.** For every `.trc` in `MarkerData` there is a
same-named `.mot` in `OpenSimData/Kinematics`. Both must be renamed identically, or the
task will fail.

Three things that are normal and need no action:

- **A task the patient did not perform** — just leave it absent. The manifest reports it
  as "missing, i.e. not attempted".
- **`neutral`** — this is the calibration pose, not a task. Leave it alone.
- **`grapevine`, `dext-left`, `dext-right`** — real tasks we do not segment. The manifest
  lists them as "out of scope".

**Bad retakes:** if a task was recorded twice and one attempt is unusable, put the word
`deleted` in that file's name — for example `tandem-deleted.trc` and
`tandem-deleted.mot`. The scripts skip anything with `delet` or `delt` in the name, and
the manifest lists it under "discarded retakes" so the decision stays visible.

---

## Part 3 — Run the scripts

### 3a. Open a terminal and switch on the environment

1. Open VS Code.
2. **Terminal → New Terminal.**
3. Switch on the environment (**every new terminal needs this**):

```
conda activate opensim_env
```

Your prompt should now start with `(opensim_env)`.

### 3b. Check you are on the right Python — do not skip this

`conda activate` is not always the last word. VS Code can auto-activate a *different*
environment for a folder, and if it does, plain `python` silently stays pointed at that
one. Check:

```
python -c "import sys; print(sys.executable)"
```

The path it prints **must contain `opensim_env`**.

If it prints anything else — a `venv`, a `qc_venv`, a bare system Python — do not fight
it. Use the full path to the right interpreter for every command from here on. Print it
once:

```
conda env list
```

Find the `opensim_env` row and copy the path on the right. Your interpreter is that path
with `/bin/python` on the end (Mac) or `\python.exe` on the end (Windows). For example:

```
/Users/yourname/miniconda3/envs/opensim_env/bin/python
```

Check it, in quotes:

```
"/Users/yourname/miniconda3/envs/opensim_env/bin/python" -c "import opensim; print('ok', opensim.__version__)"
```

You want `ok 4.6`. Wherever the rest of this SOP says `python`, paste that quoted path
instead.

> **Why not `conda run -n opensim_env python`?** It looks like it should work and it does
> not — when another environment is already active, that command still resolves to the
> other environment's Python. The full path is the only reliable route.

### 3c. Go to the scripts folder

```
cd "/Users/yourname/Desktop/OpenCapSegmentation/segmentation-scripts"
```

Substitute your own path. Doing this first means each command below carries only **one**
long path instead of two, which avoids the most common typing error (see 3d).

### 3d. Run it

**Do not type the patient folder path by hand.** Type `python run_all.py ` — including
the trailing space — then **drag the patient folder from Finder (or File Explorer) into
the terminal window**. The path types itself, correctly quoted. Then press Enter.

```
python run_all.py "PASTE_PATIENT_FOLDER_HERE"
```

> **Mind the space.** If the quote before the folder path touches the previous argument,
> the shell glues them into one and you get
> `can't open file '...run_all.py/Users/...': [Errno 20] Not a directory`. There must be
> a space between `run_all.py` and the opening quote.

**To run several patients at once,** add more folders, each in its own quotes, **each
separated by a space**:

```
python run_all.py "PATIENT_1_FOLDER" "PATIENT_2_FOLDER"
```

**To run every patient inside one parent folder,** give it the parent folder instead.

### 3e. What you will see

Text scrolls by for roughly **2–3 minutes per patient**. No windows pop up; everything is
saved to files. Each of the five steps announces itself:

```
--- manifest ---
--- spike check ---
--- segmentation ---     (~30 s of imports first)
--- marker plots ---     (~30 s of imports first)
--- coord plots ---      (~30 s of imports first)
```

The "~30 s of imports first" pauses are normal — the script is loading OpenSim and looks
frozen while it does. When everything finishes you will see:

```
========================================================================
All steps completed for all 1 session(s).
Outputs are in each session's Outputs/ folder.
```

> **Tip:** to keep a copy of everything the terminal printed, add
> `2>&1 | tee ~/Desktop/segmentation_log.txt` to the end of the command. The run log is
> the only place some warnings appear.

---

## Part 4 — Find and read the results

Everything lands in an `Outputs` folder inside the patient folder.

### `Outputs/Segmentation/events_<name>.csv` — the main deliverable

One row per event. Open it in Excel. The columns are:

| Column | Meaning |
| --- | --- |
| `task` | Which trial the event belongs to |
| `event` | The `S1`, `S2`, … label from the segmentation markers document |
| `name` | Plain-English name, e.g. "ipsilateral heel strike" |
| `cycle` | Repetition number, for tasks that repeat (gait cycles, sit-to-stand reps) |
| `anchor_leg` | For walking: which leg's heel strikes bracket this cycle |
| `side` | Which side of the body this specific event is on (`r` / `l`) |
| `frame` | Video frame number (60 frames per second) |
| `time_s` | Time in seconds from the start of the trial |
| `source` | Which file it was measured from — `trc` (markers), `mot` (angles), or `mot+model` |
| `definition` | Exactly how that instant was computed |

### `Outputs/Segmentation/run_log_<name>.txt`

The detailed record for every task: thresholds used, how the calibration trim was
decided, which events could not be produced and why, and per-cycle diagnostics. **When
something looks wrong in the CSV, the answer is almost always in here.**

### `Outputs/AnglePlots/marker_plots/`

One chart per trial showing every marker's position over time. Use these to eyeball
whether the raw marker data is sane.

### `Outputs/AnglePlots/all_coords/`

One chart per trial, 24 panels, showing every joint angle over time.

### `Outputs/AnglePlots/all_coords_events/` and `.../marker_plots` overlays

The same charts **with the segmentation events drawn on as vertical lines**. These are
the fastest way to sanity-check a result: if a heel strike is marked in the wrong place,
you will see it immediately.

---

## Part 5 — Did it work? Five acceptance checks

Scroll back through the terminal output (or your saved log) and check these five things
in order.

**1. The manifest found everything.** Look for:

```
--- UNMATCHED, data we would silently discard (0)
  none
```

If this is **not** zero, a trial name does not match the list in step 2c. Go back and
rename it. This is the check that catches the most common mistake.

**2. No task failed.** Look for:

```
NNN events across NN tasks, 0 task failures
```

If a task failed, the reason is printed right under it and repeated in the run log.

**3. The consecutiveness check passed.** Look for:

```
--- consecutiveness check
  OK: in every task, no S(N+1) lands before its S(N)
```

Anything listed here means events came out in an impossible order for that task — almost
always a sign of glitchy data in that trial.

**4. No stride intervals were flagged.** Look for:

```
--- stride interval outliers
  OK: no gait interval exceeds 2.5x the median for its event pair
```

If a cycle is listed here, the gap between two events in that walking cycle is far longer
than the same gap in every other cycle of the same trial. **Check that moment in the
video.** The usual cause is a misstep that the patient corrected by picking the foot up
and putting it down again. See the note at the end of Part 6.

**5. Spot-check one overlay.** Open one chart from
`Outputs/AnglePlots/all_coords_events/` for a task you watched being recorded, and
confirm the event lines land where you expect.

Also glance at the **spike check** near the top of the run. Trials listed there have
markers that physically teleport, and any odd segmentation result in those trials is
usually the data, not the script.

---

## Part 6 — Per-patient adjustments

### 6a. When the calibration arm raise confuses a trial

Most trials start with the patient raising both arms overhead. The scripts detect that
raise and ignore everything before it, so the calibration movement is never mistaken for
the task.

Sometimes that goes wrong — most often when the patient **starts with their arms already
raised and brings them down**, or does no raise at all. The symptom is that the beginning
of the task is missing: a reach test that reports two reaches instead of three, or a
walking trial whose first cycle starts absurdly late.

To switch the trim off for specific trials, create a plain text file called
**`no_calibration.txt`** in the patient folder (next to `MarkerData`), with **one task
name per line**:

```
frt
gait-bckw
```

Lines starting with `#` are ignored, so you can record why:

```
# started with arms already up
frt
gait-bckw
```

A single line containing `*` switches the trim off for the entire session.

Re-run Part 3. The run log will now show
`declared: no usable calibration raise on this trial` for those tasks.

### 6b. When a trial is too glitchy to use

Rename both files to contain `deleted` — `march-deleted.trc` and `march-deleted.mot` —
and re-run. The manifest will list it under "discarded retakes" so the decision is on the
record rather than invisible.

### 6c. A known limit worth understanding

A heel strike is detected as the moment the heel is furthest forward **relative to the
pelvis**. If a patient missteps and then picks the foot up and replants it, the pelvis
keeps moving forward during the replant — often faster than the foot does. When that
happens the corrected landing produces no detectable peak at all, and the event stays on
the first, abandoned contact.

The scripts do not guess at this. Instead, check 4 in Part 5 flags the cycle so a human
can look at the video and correct the time by hand. If you see a flagged cycle, expect
the true landing to be **later** than the time in the CSV.

---

## Troubleshooting

| What you see | What it means | What to do |
| --- | --- | --- |
| `conda: command not found` | Miniconda isn't installed, or VS Code was not restarted after installing. | Redo step 1b, then fully quit and reopen VS Code. |
| `PermissionError: [Errno 1] Operation not permitted`, ending in `os.getcwd()` | macOS is blocking the terminal from reading the folder it is sitting in. Conda itself is fine. | Run `cd ~` and try the command again. If it still fails, give your terminal app permission: System Settings → Privacy & Security → Full Disk Access → switch on VS Code (or Terminal), then fully quit and reopen it. |
| `ModuleNotFoundError: No module named 'opensim'` (or numpy / scipy / pandas) | You forgot to activate the environment. | Run `conda activate opensim_env` and try again. Your prompt must start with `(opensim_env)`. |
| `ModuleNotFoundError: No module named 'gait_analysis'` or `'utilsKinematics'` | `opencap-processing` is missing or in the wrong place. | Redo steps 1e–1g. Confirm step 1h prints `True`. |
| `can't open file '.../run_all.py'` | The path to the script is wrong. | `cd` into your `segmentation-scripts` folder first (step 3c), then run `python run_all.py ...`. |
| `can't open file '...run_all.py/Users/...': [Errno 20] Not a directory` | Two paths got glued together — a missing space. | There must be a space between `run_all.py` and the opening quote of the folder path. See 3d. |
| `ModuleNotFoundError: No module named 'opensim'` **even after activating** | VS Code auto-activated a different environment, so plain `python` is the wrong interpreter. | Do step 3b. Use the full `.../envs/opensim_env/bin/python` path instead of `python`. |
| `no session folder for X` | The patient folder path is wrong, or you gave a name instead of a folder. | Re-type the command and **drag the folder in** rather than typing the path. |
| A task prints `no trial in ..., skipping` | That `.trc` is absent or misnamed. | Normal if the patient didn't do it. Otherwise fix the name per step 2c. |
| `UNMATCHED` is not zero in the manifest | A trial name doesn't match our list. | Rename the `.trc` **and** its `.mot` per step 2c. |
| A task prints `FAILED -- ...` | That one trial could not be segmented. Every other task still ran. | Read the reason in the run log. Usually glitchy data; cross-check the spike check and the marker plots. |
| `PARTIAL -- not emitted: ...` | Some events of that task genuinely do not exist — e.g. the patient never returned a foot to neutral, or the recording ended mid-task. | This is correct behaviour, not an error. The named events are deliberately left out rather than guessed. |
| A login prompt for app.opencap.ai appears | Should not happen; the scripts set an offline token. | Press Ctrl-C and report it — the scripts read only local files and never need to log in. |
| A red error you don't recognise | Unknown. | Copy the full red text and Google it or ask an LLM. Include the last 20 lines of terminal output. |

---

## Appendix: which script does what

You only ever run `run_all.py`. The rest are called for you. This list is for anyone
debugging.

| Script | Role |
| --- | --- |
| `run_all.py` | The only one you run. Calls the five steps below, in order, per session. |
| `manifest.py` | What is on disk vs what we segment. |
| `flag_spikes.py` | Trials with physically impossible marker jumps. |
| `run_segmentation.py` | Runs every task module, writes the CSV, run log and overlays. |
| `plot_opencap_trc_markers.py` | Marker position charts. |
| `plot_opencap_all_coords.py` | Joint angle charts. |
| `seg_common.py` | Shared plumbing: file loading, calibration trim, turn detection, the order and interval checks. |
| `seg_walk_core.py`, `seg_sts_core.py` | Copies of the OpenCap team's own gait and sit-to-stand routines, with our documented changes. |
| `seg_<task>.py` | One module per task family, each documenting how its events are defined. |
| `plot_events_overlay.py` | Draws events onto the charts. |
| `verify_verbatim.py` | Maintainer tool: re-checks our copies of OpenCap code against the originals. Run after updating `opencap-processing`. |
| `CHANGES_FROM_OPENCAP.md` | Every deliberate difference from stock OpenCap, with the measurements behind each one. |
