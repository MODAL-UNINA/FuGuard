# FuGuard: Client-Level Federated Unlearning via Generative Surrogates and Optimal Transport

This repository supports research on the training and **unlearning** process in Federated Learning (FL). It provides a modular framework to explore and compare various Federated Unlearning (FU) methods.

## Abstract

Federated Learning (FL) is a widely adopted paradigm that enables collaborative model training while preserving data privacy. As concerns around data poisoning and the “right to be forgotten” continue to grow, \textit{federated unlearning}, which is the ability to remove the influence of specific training data from a trained FL model, has become increasingly critical. However, existing unlearning methods often require expensive retraining or high communication overhead, limiting their practicality in real-world FL systems. In this work, we propose FuGuard, a dual-strategy federated unlearning framework, designed for efficient and scalable client-level data removal. FuGuard combines generative surrogate reconstruction, which approximates the contribution of the target client, with optimal transport regularization that softly constrains model parameter drift during unlearning. This approach effectively removes the influence of the target client while preserving the stability and performance of the global model. To evaluate the forgetting capability, we conduct adversarial testing using backdoor attacks for residual data influence. Empirical results on different benchmarks demonstrate that FuGuard significantly reduces the impact of the target client’s data while maintaining the performance of non-target clients, consistently outperforming state-of-the-art baselines in both forgetting effectiveness and accuracy retention.


## Background
<p align="center">
  <img src="https://github.com/MODAL-UNINA/FuGuard/blob/main/png/backdoor.png" width="500">
</p>
Backdoor attacks in federated learning, where an attacker client trains on trigger-labeled data and uploads poisoned updates to induce malicious behavior in the global model.

## Framework
<p align="center">
  <img src="https://github.com/MODAL-UNINA/FuGuard/blob/main/png/framework.png" width="500">
</p>
Overview of the proposed framework, FuGuard. The process begins with a standard federated training phase to obtain a global model. Upon receiving a deletion request, a pre-trained generative model synthesizes a compact and privacy-preserving proxy dataset for the target client. This proxy enables a gradient ascent unlearning, where model updates are guided by an optimal transport. Finally, a few recovery steps restore the global model's performance.

## Experiment















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
   Use the `dilichlet_allocator_backdoor.py` to simulate a federated training environment with backdoor injection.

2. **Run FU Methods**  
   Select and run any of the implemented FU strategies to evaluate their performance in the folder '../code'.

---

## 📌 Requirements

```bash
pip install -r requirements.txt
