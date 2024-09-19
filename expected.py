import onnx

# Load the ONNX model and check its input shape
onnx_model_path = 'aqi_model.onnx'
onnx_model = onnx.load(onnx_model_path)

# Print input shape information
for input in onnx_model.graph.input:
    print(input.name, input.type.tensor_type.shape)
