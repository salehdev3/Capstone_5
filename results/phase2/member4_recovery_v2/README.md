# Detector ONNX export parity

The assistant executed static FP32 export at input 640, ONNX opset 17, with no NMS, using the M4 checkpoint pinned by Recovery v2. All raw-output checks passed at rtol/atol 0.001 on zero, constant and seeded-random tensors; maximum absolute difference 0.000396728515625. This is assistant CPU evidence, not a measurement of the user's Colab runtime or full-pipeline parity.

The complete segmentation pipeline still includes PyTorch M4/M5 and SAM. Neither a detector-forward benchmark nor this raw tensor comparison establishes end-to-end speed or segmentation equivalence. The original presentation's later integration cells must be updated separately to invoke recovery v2.
