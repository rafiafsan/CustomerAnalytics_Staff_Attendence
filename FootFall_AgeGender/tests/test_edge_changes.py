import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gate.event_manager import BoundedIdSet
from utils.security import mask_url
from inference.onnx_detector import ONNXDetector

assert mask_url("rtsp://admin:pw123@10.0.0.5:554/x") == "rtsp://***:***@10.0.0.5:554/x"
assert mask_url("test.avi") == "test.avi"

s = BoundedIdSet(max_size=3)
for i in range(10):
    s.add(i)
assert len(s) == 3 and 9 in s and 0 not in s

# letterbox round trip: fake detector object without loading a model
d = ONNXDetector.__new__(ONNXDetector)
d.input_width = d.input_height = 480
frame = np.full((720, 1280, 3), 200, np.uint8)
t, ratio, px, py = d.preprocess(frame)
assert t.shape == (1, 3, 480, 480) and t.dtype == np.float32
assert abs(ratio - 0.375) < 1e-6 and px == 0 and py == 105, (ratio, px, py)
# a box at native (640,360) centre -> model centre (240,240) -> back
cx = (240 - px) / ratio
cy = (240 - py) / ratio
assert abs(cx - 640) < 1 and abs(cy - 360) < 1
print("ALL OK")
