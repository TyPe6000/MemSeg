import onnx, collections, sys

onnxpath = "./export_onnx/memseg_msff_08121622.onnx"

model = onnx.load(onnxpath)

print("Loading ONNX File, Filename:", onnxpath)

supported = ["Add", "AveragePool", "BatchNormalization", "Clip", "Concat", "Conv", "ConvTranspose",
    "Constant", "DepthToSpace", "Exp", "Elu", "Erf", "Expand", "Flatten", "Gemm", "Gelu",
    "LeakyRelu", "Log", "LpNormalization", "MatMul", "MaxPool", "Mean", "Mul", "Pad",
    "ReduceL2", "ReduceSum", "Relu", "Reshape", "Resize", "Pow", "SpaceToDepth",
    "Sigmoid", "Slice", "Softmax", "Softplus", "Sub", "Split", "Sqrt",
    "Tanh", "Transpose", "Unsqueeze"
]   # Warboy 연산자 목록 https://developer.furiosa.ai/docs/latest/ko/npu/warboy.html

ops   = collections.Counter(n.op_type for n in model.graph.node)

unsupported = [k for k in ops if k not in supported]

# unsupported = [k for k in ops if k not in {
#     "Add", "AveragePool", "BatchNormalization", "Clip", "Concat", "Conv", "ConvTranspose", "Constant",
#     "DepthToSpace", "Exp", "Elu", "Erf", "Expand", "Flatten", "Gemm", "Gelu", "LeakyRelu", "Log",
#     "LpNormalization", "MatMul", "MaxPool", "Mean", "Mul", "Pad", "ReduceL2", "ReduceSum", "Relu",
#     "Reshape", "Resize", "Pow", "SpaceToDepth", "Sigmoid", "Slice", "Softmax", "Softplus",
#     "Sub", "Split", "Sqrt", "Tanh", "Transpose", "Unsqueeze"
#     }]  # Warboy 연산자 목록 https://developer.furiosa.ai/docs/latest/ko/npu/warboy.html

print("남은 미지원 op:", unsupported)