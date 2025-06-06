# FuGuard: Client-Level Federated Unlearning via Generative Surrogates and Optimal Transport

This repository supports research on the training and **unlearning** process in Federated Learning (FL). It provides a modular framework to explore and compare various Federated Unlearning (FU) methods.

## Abstract

Federated Learning (FL) is a widely adopted paradigm that enables collaborative model training while preserving data privacy. As concerns around data poisoning and the “right to be forgotten” continue to grow, \textit{federated unlearning}, which is the ability to remove the influence of specific training data from a trained FL model, has become increasingly critical. However, existing unlearning methods often require expensive retraining or high communication overhead, limiting their practicality in real-world FL systems. In this work, we propose FuGuard, a dual-strategy federated unlearning framework, designed for efficient and scalable client-level data removal. FuGuard combines generative surrogate reconstruction, which approximates the contribution of the target client, with optimal transport regularization that softly constrains model parameter drift during unlearning. This approach effectively removes the influence of the target client while preserving the stability and performance of the global model. To evaluate the forgetting capability, we conduct adversarial testing using backdoor attacks for residual data influence. Empirical results on different benchmarks demonstrate that FuGuard significantly reduces the impact of the target client’s data while maintaining the performance of non-target clients, consistently outperforming state-of-the-art baselines in both forgetting effectiveness and accuracy retention.

## Framework
<p align="center">
  <img src="https://github.com/MODAL-UNINA/FuGuard/blob/main/png/framework.png" width="600">
</p>
Overview of the proposed framework, FuGuard. The process begins with a standard federated training phase to obtain a global model. Upon receiving a deletion request, a pre-trained generative model synthesizes a compact and privacy-preserving proxy dataset for the target client. This proxy enables a gradient ascent unlearning, where model updates are guided by an optimal transport. Finally, a few recovery steps restore the global model's performance.

## Experiment
<p align="center">
  <img src="https://github.com/MODAL-UNINA/FuGuard/blob/main/png/results.png" width="600">
</p>
Using the SVHN dataset as an example, the commands below show how to reproduce the results:

```bash
python `dilichlet_allocator_backdoor.py` --dataset_name SVHN --num_clients 10 --alpha 0.1 --seed 0 --target_client f_00003 --target_label 9 --inject_ratio 1

python `../code/FuGuard.py`
--device cuda:0 --dataset SVHN --client_idx f_00003 --model ConvNet --env_path ../backdoor
--strategy dilichlet --env seed0-u10-alpha0.1 --communication_round 50 --learning_rate 0.001
--weight_decay 0.0001 --momentum 0.9 --local_epoch 5 --save_path ../save/fuguard --batch_size 256
--samples_scale 0.1 --gen_bs 64 --unlearn_bs 64 --ot_lambda 0.1 --alpha 0.5 --forgetting_epoch 5
--forgetting_lr 0.001 --recover_epoch 5 --communication_round_recover 10 --num_workers 4
--pin_memory False --persistent_workers False --seed 0

Parameter explanations:

--`device`: GPU device to use (e.g., cuda:0)

--`dataset`: Dataset name (SVHN, CIFAR10, etc.)

--`client_idx`: Target backdoor client ID

--`strategy`: Client partitioning strategy (e.g., dilichlet)

--`env`: Federated learning environment name

--`communication_round`: Number of FL communication rounds

--`samples_scale`: Sampling ratio for generation

--`gen_bs`: Batch size during generation phase

--`unlearn_bs`: Batch size during unlearning phase

--`ot_lambda`: Optimal transport unlearning weight

--`forgetting_epoch`: Epochs for forgetting step

--`forgetting_lr`: Learning rate during forgetting

--`communication_round_recover`: Communication rounds for recovery
```
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
   Select and run any of the implemented FU strategies to evaluate their performance in the folder `../code`.

---

## 📌 Requirements

```bash
pip install -r requirements.txt
