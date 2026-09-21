"""
Check that the copied OpenCap methods differ only where we say they differ.
==========================================================================

    python3 verify_verbatim.py

Compares the executable lines (comments and blank lines stripped) of our
copies against the originals in the opencap-processing checkout, and prints
every difference. Run this after pulling a new version of
opencap-processing: if OpenCap changes segment_walking or segment_sts, this
is what tells you the copies have drifted.

Expected output is the substitutions listed in EXPECTED below and nothing
else.
"""

import difflib
import os
import sys

OPENCAP = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "opencap-processing"))

# (our file, our method, original file, original method, allowed changes)
TARGETS = [
    ("seg_walk_core.py", "segment_walking",
     "ActivityAnalyses/gait_analysis.py", "segment_walking",
     ["expectedOrder = self.EXPECTED_ORDER",
      "prominences = self.PROMINENCES",
      "return self._collapse_replants(rHS,lHS,rTO,lTO)"]),
    # Both sts substitutions span several physical lines, so each is listed
    # rather than loosening the check to accept any continuation.
    ("seg_sts_core.py", "segment_sts",
     "ActivityAnalyses/sts_analysis.py", "segment_sts",
     [# 1 of 2: the lean onset itself
      "forwardLeanIdx = self._forward_lean_onset(",
      "torso_z_vel, pelvVel, int(temp_sit_ind), int(startIdx[0]),",
      "lean_threshold, velSeated)",
      # 2 of 2: the lean floor at the previous cycle's sitting frame
      "leanWindows = []",
      "leanWindows.append((int(temp_sit_ind), int(startIdx[0])))",
      "forwardLeanInds, forwardLeanTimes = self._apply_lean_floor(",
      "forwardLeanInds, startFinishIndsDelayPeriodic, leanWindows,",
      "torso_z_vel, pelvVel, timeVec, lean_threshold, velSeated)"]),
]


def code_lines(text):
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(line)
    return out


def extract(path, name):
    lines = open(path).read().split("\n")
    start = next(i for i, l in enumerate(lines)
                 if l.strip().startswith(f"def {name}("))
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = len(lines)
    for i in range(start + 1, len(lines)):
        s = lines[i]
        if s.strip() and (len(s) - len(s.lstrip())) <= indent:
            end = i
            break
    return "\n".join(lines[start:end])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    failures = 0

    for ours, our_method, theirs, their_method, allowed in TARGETS:
        our_path = os.path.join(here, ours)
        their_path = os.path.join(OPENCAP, theirs)
        if not os.path.exists(our_path):
            print(f"[SKIP] {ours} not present")
            continue
        if not os.path.exists(their_path):
            print(f"[FAIL] original not found: {their_path}")
            failures += 1
            continue

        a = code_lines(extract(their_path, their_method))
        b = code_lines(extract(our_path, our_method))
        diff = [d for d in difflib.unified_diff(a, b, lineterm="", n=0)
                if d.startswith(("+", "-"))
                and not d.startswith(("+++", "---"))]

        added = [d[1:].strip() for d in diff if d.startswith("+")]
        unexpected = [x for x in added
                      if not any(x.startswith(p) for p in allowed)]

        print(f"=== {ours}:{our_method}  vs  {theirs}:{their_method}")
        print(f"    {len(a)} executable lines original, {len(b)} ours, "
              f"{len(diff)} differing")
        for d in diff:
            print(f"      {d}")
        if unexpected:
            print(f"    [FAIL] unexpected change: {unexpected}")
            failures += 1
        else:
            print("    [OK] only the declared substitutions")
        print()

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
