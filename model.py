import onnxruntime
import numpy as np
import torch

# Load the ONNX model (optimized model)
onnx_model_path = 'fine_tuned_resnet50_optimized.onnx'
session = onnxruntime.InferenceSession(onnx_model_path)

def predict(img):
    # Preprocess image
    img = img.numpy() if torch.is_tensor(img) else np.array(img)
    img = np.expand_dims(img, axis=0)

    img = img.astype(np.float32)

    # Run the model
    inputs = {session.get_inputs()[0].name: img}
    outputs = session.run(None, inputs)

    binary_pred = outputs[0].squeeze()

    attack_pred = np.argmax(outputs[1], axis=1).item()

    return binary_pred, attack_pred