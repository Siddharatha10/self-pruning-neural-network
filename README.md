# 🧠 Self-Pruning Neural Network (Dynamic Sparsity Learning)

## 📌 Overview

This project implements a **self-pruning neural network** that learns to remove unnecessary weights during training.

Unlike traditional pruning (done after training), this model integrates pruning directly into the learning process using **learnable gate parameters**, enabling the network to dynamically decide which connections to keep or remove.

---

## 🚀 Core Idea

Each weight `w` is paired with a learnable gate `g`:

```
w_effective = w * sigmoid(g)
```

* sigmoid(g) ∈ (0,1)
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

---

### 🔹 Sparsity Loss

```
Loss = CrossEntropy + λ * SparsityLoss
```

Where:

```
SparsityLoss = mean of all gate values
```

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

```
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

---

## 📊 Visualization

### Gate Distribution

* Spike near **0** → pruned weights
* Cluster away from 0 → important weights

This confirms successful pruning during training.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* NumPy
* Matplotlib

---

## ▶️ How to Run

```
pip install -r requirements.txt
python main.py
```

---

## 📁 Project Structure

```
main.py
README.md
requirements.txt
```

---

## 🎯 Key Learnings

* Custom neural network layer design
* L1-based sparsity regularization
* Trade-off between sparsity and accuracy
* Efficient model design

---

## 🚀 Future Improvements

* Structured pruning (neuron-level)
* Hard threshold pruning
* CNN-based architecture
* Advanced gating methods

---

## 👤 Author

**Siddharatha**
M.Tech (AI/ML)
Focus: AI Engineering, ML Systems

---
