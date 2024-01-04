import numpy as np
import onnx
import onnxruntime as ort

onnx_model = onnx.load("pendulum.onnx")

# Check that the model is well formed
onnx.checker.check_model(onnx_model)

ort_model = ort.InferenceSession("pendulum.onnx")

outputs = ort_model.run(
    None,
    {"obs": np.zeros((1, 2)).astype(np.float32)},
)
print(outputs)