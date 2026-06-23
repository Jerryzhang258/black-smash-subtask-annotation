"""
Single source of truth for the ego *wipe-the-tube* annotation pipeline.

The dataset is the raw teleop **demo-folder** format (Quest 3 egocentric stereo +
bimanual grippers), NOT LeRobot parquet — so this pipeline is separate from the
black_smash (`batch_annotate.py`) code, but follows the same design: crisp
proprioceptive boundaries (gripper width + hand pose) own the geometric points,
a VLM owns the visual ones, and disagreement routes to a human.

Edit the taxonomy / thresholds here; the rest of the package imports from this file.
"""
import os

# ---------------------------------------------------------------- taxonomy ----
# 6 subtasks delimited by 5 critical points. Roles ("holder" = the hand that
# grips the tube the longest; "wiper" = the other hand) are assigned per demo by
# grasp duration, so this is robust to left/right-handed demos.
LABELS = [
    "reach for the test tube",                      # S0: 0          .. c1
    "grasp and lift the test tube",                 # S1: c1 (grasp) .. c2
    "acquire the wiper and bring it to the tube",   # S2: c2 (grasp) .. w
    "wipe the test tube",                           # S3: w  (motion).. c3
    "place the test tube back",                     # S4: c3 (release).. c4
    "release and retract",                          # S5: c4 (release).. end
]

# 5 critical points = start frames of S1..S5, each owned by the method that nails it.
CRIT_NAMES = ["grasp_tube", "acquire_wiper", "start_wipe", "finish_wipe", "release_tube"]
CRIT_OWNER = ["state", "state", "state", "state", "state"]   # which modality owns each
#                grasp   acquire  wipe    finish   release
# (start_wipe is the one boundary the VLM is expected to help most with; the four
#  gripper events are crisp in the width signal — see fuse.py for ownership.)

# --------------------------------------------------------------- thresholds ---
GRIP_CLOSE_FRAC = 0.45   # "closed" = normalized width below (rest - this*range)
GRIP_GAP_S      = 0.30   # morphological gap-close on the closed mask (seconds)
GRIP_MIN_HOLD_S = 0.30   # reject a closed run shorter than this as noise

RAW_SMOOTH_S    = 0.15   # per-frame speed smoothing window (seconds)
CARRIER_SMOOTH_S = 0.60  # rolling-mean window defining slow "carrier" pose
WIPE_RAW_FRAC   = 0.15   # wipe = raw speed above this fraction of its in-window max
WIPE_DRIFT_PCT  = 45     # ...AND carrier drift below this percentile (in place)
WIPE_GAP_S      = 0.50   # gap-close on the wipe mask

DEFAULT_FPS     = 60.0   # fallback if per-frame timestamps are unavailable

# ------------------------------------------------------------------- paths ----
# Defaults live under the repo (drop the demo folders into <repo>/demos/).
# Override with --demos/--out or the env vars. (data dirs are .gitignored.)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMOS_ROOT = os.environ.get("EGO_WIPE_DEMOS", os.path.join(_REPO, "demos"))
OUT_ROOT   = os.environ.get("EGO_WIPE_OUT",   os.path.join(_REPO, "wipe_annotations"))

# Per-frame frames used for visualization and VLM input. NOTE: this is NOT the
# Quest headset ego video (that is ego_data/<id>/*.mp4 — a 1280^2, ~73deg-FOV,
# HEVC "spatialmp4" that needs special decoding). It is the per-HAND wide-angle
# *fisheye* camera (left_hand_visual_img / right_hand_visual_img, 224^2, ~180deg,
# circular vignette), which sees both hands + the tube every frame and needs no
# video decode — a better visual input here. Segmentation does not use any camera.
FRAME_SOURCE = os.environ.get("EGO_WIPE_FRAMES", "hand")   # "hand" | "ego"

# "hand": per-hand fisheye jpgs (default; no decode needed)
FRAME_DIR  = "left_hand_visual_img"      # or "right_hand_visual_img"
FRAME_FMT  = "left_hand_{i}.jpg"         # match the chosen dir's filename stem

# "ego": the Quest headset stereo mp4 (2560x1280 side-by-side HEVC). Frames are
# decoded with pyav and the chosen eye is cropped out.
EGO_VIDEO_GLOB = "ego_data/*.mp4"
EGO_EYE        = "left"                   # "left" | "right"

# ego(~30 fps) <-> hand(~60 fps) timestamp alignment. The headset and the robot
# run on DIFFERENT, unsynchronized clocks (observed offset ~minutes), so absolute
# unix sync is not reliable. We align by ELAPSED time from each stream's first
# frame, using each stream's real per-frame timestamps:
#   hand: tag_detection time (unix s, frame-aligned)   ego: camera_frames timestamp_ns
# This handles the different frame rates exactly; it assumes a common physical
# START. EGO_TIME_OFFSET_S nudges ego later(+)/earlier(-) if starts differ.
#   "elapsed"      - timestamp elapsed-time matching (default, recommended)
#   "proportional" - normalize each stream's timestamps to [0,1] then match
#                    (assumes common start AND end; fallback when timestamps missing)
EGO_ALIGN        = "elapsed"
EGO_TIME_OFFSET_S = 0.0
