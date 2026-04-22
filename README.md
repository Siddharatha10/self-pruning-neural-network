# 🧠 Self-Pruning Neural Network (Dynamic Sparsity Learning)

## 📌 Overview

This project implements a **self-pruning neural network** that learns to remove unnecessary weights during training.

Unlike traditional pruning (done after training), this model integrates pruning directly into the learning process using **learnable gate parameters**, enabling the network to dynamically decide which connections to keep or remove.

---

## 🚀 Core Idea

Each weight ( w ) is paired with a learnable gate ( g ):

[
w_{effective} = w \cdot \sigma(g)
]

* ( \sigma(g) \in (0,1) ) using sigmoid
* Gate ≈ 1 → connection active
* Gate ≈ 0 → connection pruned

---

## ⚙️ Methodology

### 🔹 Prunable Layer

* Custom `PrunableLinear` layer
* Contains:

  * weight
  * bias
  * gate_scores (learnable)

### 🔹 Sparsity Loss

[
\text{Loss} = \text{CrossEntropy} + \lambda \cdot \text{SparsityLoss}
]

Where:

[
\text{SparsityLoss} = \text{mean of all gate values}
]

* Encourages many gates → 0
* Produces a sparse network

---

## 🧠 Why L1 Regularization Works

* L1 pushes values toward **exact zero**
* Since gates are positive, minimizing their sum forces many connections to vanish
* This enables **automatic pruning during training**

---

## 📊 Dataset

* CIFAR-10 (10-class image classification)
* Used subset for faster experimentation

---

## 🏗️ Model Architecture

```id="f1w0kq"
Input (32×32×3)
   ↓
PrunableLinear (256) + ReLU
   ↓
PrunableLinear (128) + ReLU
   ↓
PrunableLinear (10)
   ↓
Output (logits)
```

---

## 📈 Results

| Lambda | Accuracy | Sparsity (%) |
| ------ | -------- | ------------ |
| 1e-4   | ~0.55    | Low          |
| 5e-4   | ~0.50    | Medium       |
| 1e-3   | ~0.45    | High         |

### 🔍 Observations

* Increasing λ increases sparsity
* Higher sparsity reduces accuracy
* There is a clear **trade-off between efficiency and performance**

---

## 📊 Visualization

### Gate Distribution

* Spike near **0** → pruned weights
* Cluster away from 0 → important weights

This confirms that the network successfully learns which connections are unnecessary.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* NumPy
* Matplotlib

---

## ▶️ How to Run

```bash id="m7p1d2"
pip install torch torchvision numpy matplotlib
python main.py
```

---

## 📁 Project Structure

```id="c4v3z1"
├── main.py
├── README.md
├── requirements.txt
```

---

## 🎯 Key Learnings

* Implementing custom neural network layers
* Applying L1-based sparsity regularization
* Understanding pruning vs accuracy trade-offs
* Designing efficient deep learning systems

---

## 🚀 Future Improvements

* Structured pruning (neuron/channel level)
* Hard threshold pruning after training
* CNN-based architecture for better accuracy
* Advanced gating methods (e.g., Gumbel-Softmax)

---

## 👤 Author

**Siddharatha**
M.Tech (AI/ML)
Focus: AI Engineering, ML Systems, LLM Applications
