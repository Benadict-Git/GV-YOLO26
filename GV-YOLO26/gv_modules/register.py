"""Make the GV-YOLO26 layers usable from Ultralytics YAML model files.

Ultralytics keeps the lists of known layer types inside `parse_model()`, so they cannot be extended from outside. This
module rewrites that function once at import time to add our layers to the right lists, and fails loudly if the
installed Ultralytics version is laid out differently from the one this project was written for.
"""

import inspect

import ultralytics
import ultralytics.nn.tasks as tasks

from .blocks import GSConv, VoVGSCSPC
from .head import LGDH

TESTED_VERSION = "8.4.156"


def _patch_source(src):
    # base_modules: (c1, c2, ...) with c2 scaled by width; repeat_modules: depth-scaled n inserted as 3rd arg;
    # detect heads: reg_max, end2end and input channels appended
    base_end = "            A2C2f,\n        }\n    )\n    repeat_modules"
    repeat_end = "            A2C2f,\n        }\n    )\n    for i, (f, n, m, args)"
    detect_end = "                OBB26,\n            }\n        ):\n            args.extend([reg_max, end2end"
    for anchor in (base_end, repeat_end, detect_end):
        if src.count(anchor) != 1:
            raise RuntimeError(
                f"gv_modules: Ultralytics {ultralytics.__version__} is not supported "
                f"(parse_model layout changed; tested with {TESTED_VERSION})."
            )
    src = src.replace(base_end, base_end.replace("A2C2f,\n", "A2C2f,\n            GSConv,\n            VoVGSCSPC,\n"))
    src = src.replace(repeat_end, repeat_end.replace("A2C2f,\n", "A2C2f,\n            VoVGSCSPC,\n"))
    src = src.replace(detect_end, detect_end.replace("OBB26,\n", "OBB26,\n                LGDH,\n"))
    return src


def register():
    if getattr(tasks, "_gv_registered", False):
        return
    for name, obj in {"GSConv": GSConv, "VoVGSCSPC": VoVGSCSPC, "LGDH": LGDH}.items():
        setattr(tasks, name, obj)  # parse_model resolves layer names through the tasks module globals
    src = _patch_source(inspect.getsource(tasks.parse_model))
    exec(compile(src, tasks.__file__, "exec"), tasks.__dict__)
    tasks._gv_registered = True
