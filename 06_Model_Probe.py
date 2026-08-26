# Model effectiveness probe -- a measuring instrument, not the service.
#
# Polls every camera in cameras.json in turn, runs BOTH detection models
# on each frame (the dedicated plate model and the truck/door model),
# records what each one saw, and saves an annotated image whenever either
# finds something. Answers "which model actually works, at which bays,
# and how often" from evidence instead of from spot-checking logs.
#
# Touches nothing the service owns -- no bay state, no sessions, no ALPR
# reads, no enter/leave results -- so it is safe to run alongside a live
# service or on its own.
#
# Two deliberate differences from the production pipeline, both because
# matching it would corrupt the measurement:
#
#   * No geometry filters. The ALPR pipeline drops boxes for too_small /
#     upper_half / off_center before OCR, and in the field that discarded
#     ALL 11 boxes on a bay where the plate model was working perfectly.
#     A probe applying the same filters would report zero and hide the
#     one thing it exists to find. Raw model output only.
#   * Full frame, not the ROI (model_probe.use_roi to compare). A
#     mis-placed ROI looks identical to a blind model when you only ever
#     see the filtered result.
#
# Run it:   python 06_Model_Probe.py            (Ctrl-C to stop)
#
# Reads the same config.json and cameras.json as the service, from the
# "model_probe" config block. Writes to audit/model_probe/:
#
#   detections.jsonl   one JSON object per frame -- the analysis source
#                      of truth, both models' full raw output
#   summary.json       running totals, per bay and overall, rewritten
#                      as it goes so it is readable mid-run
#   <bay>/*.jpg        annotated frames; the filename carries the
#                      verdict (p<plate-model> t<trucks> tp<truck-model
#                      plates>), so a folder listing alone shows which
#                      model found what
#
# Box colours on the saved images: GREEN = the dedicated plate model,
# BLUE = the truck model's truck classes, YELLOW = the truck model's own
# Number_Plate class. Where green and yellow disagree is the direct A/B
# between the two on plate detection.
#
# The log prints a periodic SUMMARY with the agreement counts that settle
# the question -- how often each model sees something the other misses --
# and one final summary on Ctrl-C.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from alpr_service.probe_main import main

if __name__ == "__main__":
    main()
