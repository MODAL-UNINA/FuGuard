# Federated Unlearning (FU) Experiments

This repository supports research on the training and **unlearning** process in Federated Learning (FL). It provides a modular framework to explore and compare various Federated Unlearning (FU) methods.

---

## 📁 Project Structure

### 1. FL Environment Setup

- **`dilichlet_allocator_backdoor`**  
  Used to generate federated learning environments. Supports various data distributions:
  - Dirichlet (non-IID)
  - Backdoor injection

### 2. FU Methods

Implemented Federated Unlearning methods:

- **Retrain**  
  A naive retraining strategy on both the server and client side.

- **FedSGA**  
  Implements Stochastic Gradient Ascent for server and client unlearning.

- **QuickDrop**  
  Based on the paper:  
  _"QuickDrop: Efficient Federated Unlearning via Synthetic Data Generation"_

- **FU**  
  Based on the paper:  
  _"Federated Unlearning: How to Efficiently Erase a Client in FL?"_

- **FuGuard**  
  Our proposed method for efficient and privacy-preserving client unlearning.

---

## 🚀 How to Use

1. **Generate FL Environment**  
   Use the `dilichlet_allocator_backdoor` module to simulate a federated training environment.

2. **Run FU Methods**  
   Select and run any of the implemented FU strategies to evaluate their performance.

---

## 📌 Requirements

```bash
pip install -r requirements.txt
