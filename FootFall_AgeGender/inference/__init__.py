from .detector import create_detector
from .onnx_detector import ONNXDetector, ONNX_AVAILABLE, OPENVINO_AVAILABLE
from .inference_scheduler import InferenceScheduler


def __getattr__(name):
    # Lazy: importing TorchDetector pulls in torch + ultralytics (heavy, optional on the edge PC)
    if name == "TorchDetector":
        from .torch_detector import TorchDetector

        return TorchDetector
    raise AttributeError(name)


__all__ = ["create_detector", "ONNXDetector", "TorchDetector", "InferenceScheduler", "ONNX_AVAILABLE", "OPENVINO_AVAILABLE"]
