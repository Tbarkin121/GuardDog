# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:38:17 2024

@author: tylerbarkin
"""

import numpy as np
import onnx
import onnxruntime as ort

onnx_model = onnx.load("pendulum.onnx")

# Check that the model is well formed
onnx.checker.check_model(onnx_model)

ort_model = ort.InferenceSession("pendulum.onnx")

in_data = np.zeros((1,2))
in_data[0,0] = 0.52
in_data[0,1] = -0.32
outputs = ort_model.run(
    None,
    {"obs": in_data.astype(np.float32)},
)
print(outputs)

#Results from STM32 with 0.52 and -0.32 as inputs
# 1.80677, -0.67291, 0.82512

# Results from python with 0.52 and -0.32 as inputs
# 1.80461, -0.67291, 0.82535

# Minor differences due to quantization and compression? The more compression the worse the inference does 
