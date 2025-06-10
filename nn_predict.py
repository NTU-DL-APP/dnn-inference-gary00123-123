import numpy as np
import json

# === Activation functions ===
def relu(x):
    # Rectified Linear Unit: max(0, x)
    return np.maximum(0, x)

def softmax(x):
    # Numerically stable softmax
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b  # 等同於 np.dot(x, W) + b

# === Forward pass using parsed architecture ===
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x

# === Main inference function ===
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
