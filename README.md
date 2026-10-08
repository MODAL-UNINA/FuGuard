# FuGuard: Client-Level Federated Unlearning via Generative Surrogates and Optimal Transport

This repository supports research on the training and **unlearning** process in Federated Learning (FL). It provides a modular framework to explore and compare various Federated Unlearning (FU) methods.

The paper details are as follows:
P. Qi, D. Annunziata, C. Jappelli, F. Giampaolo and F. Piccialli, "FuGuard: Client-Level Federated Unlearning via Generative Surrogates and Optimal Transport," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2026.3708982. [Paper](https://ieeexplore.ieee.org/abstract/document/11604159)

## Abstract

Federated Learning (FL) is a widely adopted paradigm that enables collaborative model training while preserving data privacy. As concerns around data poisoning and the “right to be forgotten” continue to grow, federated unlearning, which is the ability to remove the influence of specific training data from a trained FL model, has become increasingly critical. However, existing unlearning methods often require expensive retraining or fail to achieve good forgetting effects, limiting their practicality in real-world FL systems. In this work, we propose FuGuard, a dual-strategy federated unlearning framework, designed for efficient and ideal client-level data removal. FuGuard combines the generative surrogate, which approximates the contribution of the target client, with optimal transport regularization that softly constrains model parameter drift during unlearning. This approach effectively removes the influence of the target client while preserving the stability and performance of the global model. To evaluate the forgetting capability, we conduct testing using backdoor attacks and member inference attacks (MIA) for residual data influence. Empirical results on different benchmarks demonstrate that FuGuard significantly reduces the impact of the target client’s data while maintaining the performance of non-target clients, consistently outperforming state-of-the-art baselines in both forgetting effectiveness and accuracy retention. Our code is accessible at:
\url{https://anonymous.4open.science/r/FuGuard-0263}.

## Framework
<p align="center">
  <img src="https://github.com/MODAL-UNINA/FuGuard/blob/main/png/framework.png" width="800">
</p>
Overview of the proposed framework, FuGuard. The process begins with a standard federated training phase to obtain a global model. Upon receiving a deletion request, a pre-trained generative model synthesizes a compact and privacy-preserving proxy dataset for the target client. This proxy enables a gradient ascent unlearning, where model updates are guided by an optimal transport. Finally, a few recovery steps restore the global model's performance.

## Experiment
<p align="center">
  <img src="https://github.com/MODAL-UNINA/FuGuard/blob/main/png/results_table.png" width="600">
</p>



## 🚀 How to Use

### 1. FL data preparation

- **`prepare_data`**  
  Used to generate federated learning client data. Supports various data distributions:
  - Dirichlet (non-IID)
  - Backdoor injection

### 2. FU Methods

Implemented Federated Unlearning methods:

- **Retrain**  
  A naive retraining strategy on both the server and client side.

- **FedSGA**  
  Implements Stochastic Gradient Ascent for server and client unlearning.

- **Fast-FedUL**  
  Based on the paper:  
  _"Fast-fedul: A training-free federated unlearning with provable skew resilience"_

- **FU**  
  Based on the paper:  
  _"Federated Unlearning: How to Efficiently Erase a Client in FL?"_

- **NoT**  
  Based on the paper:  
  _"NoT: Federated Unlearning via Weight Negation"_
  
- **FuGuard**  
  Our proposed method for efficient and privacy-preserving client unlearning.

---

## 📌 Requirements

```bash
pip install -r requirements.txt
