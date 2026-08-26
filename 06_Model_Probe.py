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
#   images/*.jpg       annotated frames from every bay, in ONE flat
#                      folder, named
#                        <bay>_<timestamp>_p<plate-model boxes>
#                        _t<trucks>_tp<truck-model plates>.jpg
#                      Bay and time lead, so one sorted listing groups
#                      each camera chronologically; the verdict trails,
#                      so the listing alone shows which model found what
#                      without opening anything.
#
# Box colours on the saved images: GREEN = the dedicated plate model,
# BLUE = the truck model's truck classes, YELLOW = the truck model's own
# Number_Plate class. Where green and yellow disagree is the direct A/B
# between the two on plate detection.
#
# By default only frames where the truck model reported Truck_Enter_Closed
# or Truck_Enter_Open get an image (model_probe.save_images="classes" +
# save_classes): entry is the only moment a dock camera can read a plate,
# so a run full of docked-truck frames is mostly noise when that is the
# question. detections.jsonl still records EVERY frame from BOTH models
# regardless, so narrowing what gets saved never costs measurement data.
# Class alone isn't enough, though: clearly DOCKED trucks come back as
# Truck_Enter_Open/Closed. A single-frame geometric rule settles it --
# a valid entry's box must not reach further down the frame than
# model_probe.enter_max_bottom_frac (3/4 by default). A truck that has
# finished reversing in sits against the dock, and so against the bottom
# of the frame, pushing its box past that line; one still approaching
# ends higher up. Every box's bottom_frac is recorded either way, so the
# threshold can be re-picked from real data.
#
# Other save_images modes: "any", "plate", "truck", "all", "none".
#
# The log prints a periodic SUMMARY with the agreement counts that settle
# the question -- how often each model sees something the other misses --
# and one final summary on Ctrl-C.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from alpr_service.probe_main import main

if __name__ == "__main__":
    main()
