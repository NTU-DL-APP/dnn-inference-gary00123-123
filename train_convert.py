import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 自動建立 model 資料夾
os.makedirs("model", exist_ok=True)

# === 1. 載入與標準化資料 ===
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# === 2. 建立強化版神經網路模型 ===
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(384, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# === 3. 編譯與訓練 ===
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=25, batch_size=128, validation_split=0.1)

# === 4. 評估準確率 ===
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# === 5. 儲存模型架構與權重 ===
model_path = "model/fashion_mnist"

# 5-1 儲存權重（轉成 npz）
weights = {}
for i, layer in enumerate(model.layers):
    if isinstance(layer, layers.Dense):
        w, b = layer.get_weights()
        weights[f'dense_{i}_kernel:0'] = w
        weights[f'dense_{i}_bias:0'] = b
np.savez(f"{model_path}.npz", **weights)

# 5-2 儲存模型架構（轉成 json）
arch = []
for i, layer in enumerate(model.layers):
    if isinstance(layer, layers.Dense):
        arch.append({
            "name": f"dense_{i}",
            "type": "Dense",
            "config": {"activation": layer.activation.__name__},
            "weights": [f'dense_{i}_kernel:0', f'dense_{i}_bias:0']
        })

with open(f"{model_path}.json", "w") as f:
    json.dump(arch, f, indent=2)
