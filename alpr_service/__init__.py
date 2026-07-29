"""ALPR MQTT service, split into independently-editable modules.

config.py        -- config.json loading, validation, defaults
logging_setup.py -- configurable, stage-tagged logging
cameras.py       -- cameras.json loading/validation
audit.py         -- audit folder retention pruning
image_ops.py     -- crop prep, sharpness/duplicate checks, frame-size QA
plate_text.py    -- plate normalization/validation/OCR-error-correction/voting
snapshot.py      -- HTTP snapshot capture (digest auth, camera's /snap.jpg)
mqtt_bus.py       -- MQTT event parsing and the enqueue/cooldown debounce
worker.py         -- per-job pipeline (capture -> detect -> OCR -> vote -> publish)
service.py        -- process entry point that wires everything together
"""
