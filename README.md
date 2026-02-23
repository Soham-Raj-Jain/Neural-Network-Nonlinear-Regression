# Neural-Network-Nonlinear-Regression
> **3-Layer Deep Neural Networks** for nonlinear regression using NumPy, PyTorch, PyTorch Lightning, and TensorFlow (4 variants).
> **Equation:** `y = sin(x₁) · cos(x₂) + x₃² + noise` — 3-variable synthetic dataset

---

## 📁 Repository Structure

```
📦 deep-neural-network-nonlinear-regression/
│
├── 📓 colab_a_numpy_scratch.ipynb          ← NumPy + tf.einsum, manual backprop
├── 📓 colab_b_pytorch_scratch.ipynb        ← PyTorch raw tensors, no nn.Module
├── 📓 colab_c_pytorch_classes.ipynb        ← PyTorch nn.Module + Adam
├── 📓 colab_d_pytorch_lightning.ipynb      ← PyTorch Lightning Trainer
├── 📓 colab_e_tensorflow_variants.ipynb    ← TF: 4 variants (scratch→high-level)
└── 📄 README.md                            ← This file
```

---

## 🎥 Video Walkthroughs (REQUIRED)

> Each video walks through the full Colab notebook: code sections, training output, and final plots.

| Colab | Description | Video Link |
|-------|-------------|------------|
| **A** | NumPy Scratch + tf.einsum + Manual Backprop | 📹 [Watch Colab A Walkthrough](#) |
| **B** | PyTorch Scratch (no nn.Module) | 📹 [Watch Colab B Walkthrough](#) |
| **C** | PyTorch Classes (nn.Module) | 📹 [Watch Colab C Walkthrough](#) |
| **D** | PyTorch Lightning | 📹 [Watch Colab D Walkthrough](#) |
| **E** | TensorFlow All 4 Variants | 📹 [Watch Colab E Walkthrough](#) |

> 🔔 **Note:** Replace `#` links with your actual YouTube/Loom/Drive video URLs after recording.

---

## 📓 Colab Notebooks — Detailed Explanations

---

### 📓 Colab A — NumPy From-Scratch with `tf.einsum`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_a_numpy_scratch.ipynb)

**File:** `colab_a_numpy_scratch.ipynb`

#### What It Does
Implements a **3-layer neural network entirely in NumPy** with **manual backpropagation** using the chain rule. Matrix multiplications are performed using **`tf.einsum`** instead of `np.dot`.

#### Architecture
```
Input (3) → Hidden1 (16, Swish) → Hidden2 (8, Swish) → Output (1, Linear)
```

#### Key Code Sections

| Section | Description |
|---------|-------------|
| **Data Generation** | 3-variable nonlinear: `y = sin(x1)*cos(x2) + x3²`, N=1000 samples |
| **4D Visualization** | PCA reduces 3 inputs → 2D, 3D scatter with color = target value |
| **tf.einsum** | `tf.einsum('ij,jk->ik', A, W)` replaces all matrix multiplies |
| **Swish Activation** | `z * sigmoid(z)` and its analytical derivative for backprop |
| **Forward Pass** | 3 sequential layer transformations, cache for backprop |
| **Manual Backprop** | Chain rule: `dZ3 → dW3 → dZ2 → dW2 → dZ1 → dW1` |
| **SGD Update** | `W -= lr * dW` for all 6 parameter tensors |
| **Loss Curve** | Log-scale MSE over 2000 epochs |
| **True vs Pred** | Scatter plot showing regression fit quality |

#### Notable Design Choices
- **Swish** (not ReLU or Sigmoid): smooth nonlinearity, works well for regression
- **He Initialization**: `std = sqrt(2/fan_in)` prevents vanishing gradients
- **tf.einsum notation**: `'ij,jk->ik'` = batched matrix multiply (N samples × features)

---

### 📓 Colab B — PyTorch From Scratch (No `nn.Module`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_b_pytorch_scratch.ipynb)

**File:** `colab_b_pytorch_scratch.ipynb`

#### What It Does
Pure PyTorch with **raw `torch.Tensor`** weights. No `nn.Linear`, no `nn.Module`, no optimizer class — just tensors with `requires_grad=True` and manual SGD.

#### Architecture
```
Input (3) → Hidden1 (16, Swish) → Hidden2 (8, Swish) → Output (1, Linear)
```

#### Key Code Sections

| Section | Description |
|---------|-------------|
| **Weight Init** | `make_weight()` returns `requires_grad=True` tensors with He init |
| **Swish** | `z * torch.sigmoid(z)` — pure tensor operation |
| **Forward Pass** | `X @ W1 + b1` — standard matmul, no layer abstraction |
| **Autograd** | `loss.backward()` fills `.grad` on all parameter tensors |
| **Manual SGD** | `with torch.no_grad(): p -= lr * p.grad; p.grad.zero_()` |
| **Training Loop** | 2000 epochs, print every 200 |

#### Why No `nn.Module`?
- Forces understanding of what `nn.Linear` does internally
- Shows that PyTorch autograd works on ANY tensor graph
- Demonstrates gradient flow manually

---

### 📓 Colab C — PyTorch Classes (`nn.Module`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_c_pytorch_classes.ipynb)

**File:** `colab_c_pytorch_classes.ipynb`

#### What It Does
Production-style PyTorch using `nn.Module`, `nn.Linear`, `torch.optim.Adam`, and `LR scheduler`.

#### Architecture
```
Input (3) → Dense(16) + Swish → Dense(8) + Swish → Dense(1)
```

#### Key Code Sections

| Section | Description |
|---------|-------------|
| **Custom Swish Module** | `class Swish(nn.Module)` — reusable activation layer |
| **ThreeLayerNet** | `__init__` defines layers; `forward()` chains them |
| **He Initialization** | `nn.init.kaiming_normal_` applied to all linear layers |
| **Adam Optimizer** | `optim.Adam(model.parameters(), lr=0.01)` |
| **LR Scheduler** | `StepLR(step_size=500, gamma=0.5)` — halves LR every 500 epochs |
| **Train Loop** | `zero_grad → forward → loss → backward → step → scheduler.step` |
| **Model Summary** | Total parameter count printed |

---

### 📓 Colab D — PyTorch Lightning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_d_pytorch_lightning.ipynb)

**File:** `colab_d_pytorch_lightning.ipynb`

#### What It Does
Wraps the same 3-layer net in **`LightningModule`** for structured training with automatic logging, early stopping, and LR scheduling.

#### Architecture
```
Input (3) → Dense(16) + Swish → Dense(8) + Swish → Dense(1)
```

#### Key Code Sections

| Section | Description |
|---------|-------------|
| **LightningThreeLayerNet** | Extends `L.LightningModule`; defines `net = nn.Sequential(...)` |
| **training_step** | Called per batch: forward → loss → `self.log('train_loss')` |
| **validation_step** | Called on val batches: logs `val_loss` |
| **configure_optimizers** | Returns Adam + StepLR scheduler |
| **DataLoader** | 80/20 train/val split using `TensorDataset` |
| **EarlyStopping** | Stops if `val_loss` doesn't improve for 100 epochs |
| **Trainer** | `L.Trainer(max_epochs=500, callbacks=[early_stop])` |

#### Lightning vs Plain PyTorch
| Feature | Plain PyTorch | Lightning |
|---------|---------------|-----------|
| Train loop | Manual | Automatic |
| Device handling | Manual `.to(device)` | Automatic |
| Logging | Print | `self.log()` → TensorBoard |
| Early stopping | Manual | Callback |

---

### 📓 Colab E — TensorFlow: All 4 Variants

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_e_tensorflow_variants.ipynb)

**File:** `colab_e_tensorflow_variants.ipynb`

One notebook containing **4 progressively higher-level TF implementations**:

---

#### E-i: TF From Scratch (GradientTape + raw Variables)

| Section | Description |
|---------|-------------|
| **he_var()** | Creates `tf.Variable` with He initialization |
| **forward_raw()** | Uses `tf.einsum` for each layer matmul |
| **GradientTape** | Records operations, `tape.gradient(loss, raw_params)` computes grads |
| **apply_gradients** | Adam optimizer step on raw variable list |

```python
with tf.GradientTape() as tape:
    y_pred = forward_raw(X_tf)
    loss = tf.reduce_mean((y_pred - Y_tf)**2)
grads = tape.gradient(loss, raw_params)
optimizer.apply_gradients(zip(grads, raw_params))
```

---

#### E-ii: TF Sequential with Built-in Dense Layers

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, kernel_initializer='he_normal'),
    SwishLayer(),
    tf.keras.layers.Dense(8),
    SwishLayer(),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, batch_size=64, validation_split=0.2)
```

---

#### E-iii: TF Functional API

```python
inputs = tf.keras.Input(shape=(3,))
x = SwishLayer()(tf.keras.layers.Dense(16)(inputs))
x = SwishLayer()(tf.keras.layers.Dense(8)(x))
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```
Best for: **multi-input/output models**, **shared layers**, **DAG architectures**

---

#### E-iv: TF Subclassed Model (High-Level)

```python
class ThreeLayerTF(tf.keras.Model):
    def __init__(self):
        self.fc1 = Dense(16); self.fc2 = Dense(8); self.fc3 = Dense(1)
    def call(self, x):
        x = swish(self.fc1(x)); x = swish(self.fc2(x)); return self.fc3(x)
```
Adds: **EarlyStopping** + **ReduceLROnPlateau** callbacks.

---

## 🔬 Shared Dataset Details

**Equation:**
```
y = sin(x₁) · cos(x₂) + x₃² + ε,   ε ~ N(0, 0.1)
```

| Variable | Range |
|----------|-------|
| x₁ | [-π, π] |
| x₂ | [-π, π] |
| x₃ | [-2, 2] |
| N (samples) | 1000 |

**4D Visualization Method:**
- Use `sklearn.decomposition.PCA` to reduce 3 input dimensions → 2 principal components
- Plot PCA1, PCA2, y on 3D axes; color-encode y for 4th dimension
- Explained variance printed for transparency

---

## 🏗️ Common Architecture Across All Colabs

```
         ┌──────────┐
Input    │  x1      │
(3D) ────│  x2      │────►  Dense(16) ──Swish──►  Dense(8) ──Swish──►  Dense(1) ──► ŷ
         │  x3      │
         └──────────┘
```

| Layer | Neurons | Activation |
|-------|---------|------------|
| Input | 3 | — |
| Hidden 1 | 16 | Swish |
| Hidden 2 | 8 | Swish |
| Output | 1 | Linear |

**Why Swish?**
- `f(x) = x · σ(x)` — smooth, non-monotonic
- Outperforms ReLU on many regression tasks
- Differentiable everywhere (needed for clean backprop)

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
Click any **"Open in Colab"** badge above → Runtime → Run All

### Option 2: Local
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install numpy tensorflow torch lightning scikit-learn matplotlib
jupyter notebook colab_a_numpy_scratch.ipynb
```

---

## 📊 Expected Results

| Colab | Framework | Final Val MSE |
|-------|-----------|---------------|
| A | NumPy + tf.einsum | ~0.05–0.15 |
| B | PyTorch Scratch | ~0.05–0.15 |
| C | PyTorch nn.Module | ~0.03–0.10 |
| D | PyTorch Lightning | ~0.03–0.10 |
| E-i | TF GradientTape | ~0.03–0.10 |
| E-ii | TF Sequential | ~0.02–0.08 |
| E-iii | TF Functional | ~0.02–0.08 |
| E-iv | TF Subclassed | ~0.02–0.08 |

---

## 📚 Key Concepts Demonstrated

| Concept | Where |
|---------|-------|
| Manual backprop / chain rule | Colab A |
| tf.einsum for matmul | Colab A, E-i |
| PyTorch autograd on raw tensors | Colab B |
| nn.Module OOP design | Colab C |
| Lightning structured training | Colab D |
| GradientTape low-level TF | Colab E-i |
| Keras Sequential API | Colab E-ii |
| Keras Functional API | Colab E-iii |
| Keras Model subclassing | Colab E-iv |
| PCA for 4D visualization | Colab A |
| He initialization | All |
| Swish activation | All |

---

## 👤 Author

**[Your Name]** — [Your Institution / Course Name]

> Built for: Deep Learning Assignment — 3-Layer Neural Network Nonlinear Regression
