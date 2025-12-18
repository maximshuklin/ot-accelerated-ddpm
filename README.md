# Diffusion Model Acceleration using Optimal Transport

This project explores **accelerating diffusion models** by truncating the diffusion process and learning an efficient way to sample from intermediate diffusion states using **Optimal Transport (OT)**.

---

## Introduction

Diffusion models have emerged as a powerful class of generative models, achieving state-of-the-art performance in image, audio, and data generation. However, their main drawback is **slow inference**, as generation typically requires hundreds or thousands of iterative denoising steps.

This project investigates how **Optimal Transport** can be used to reduce the number of diffusion steps while preserving sample quality.

---

## Background: Diffusion Models

### Forward (Diffusion) Process

A standard diffusion model defines a **forward noising process** that gradually transforms data into Gaussian noise. Given a data sample $x_0 \sim p_\text{data}$, the forward process is defined as:

$$
q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I), \quad t = 1, \dots, T
$$

With a suitable noise schedule ${\beta_t}_{t=1}^T$, this process admits a closed-form expression:

$$
q(x_t | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)
$$

where $\bar{\alpha}*t = \prod*{s=1}^t (1 - \beta_s)$. As $t \to T$ (typically $T \approx 1000$), the distribution of $x_T$ approaches a standard Gaussian:

$$
q(x_T) \approx \mathcal{N}(0, I)
$$

---

### Reverse (Denoising) Process

The goal of a diffusion model is to **learn the reverse process**:

$$
p_\theta(x_{t-1} | x_t)
$$

which iteratively removes noise from $x_T$ to recover a sample from the data distribution. This is done by training a neural network (usually a U-Net) to predict either the noise $\varepsilon$, the clean sample $x_0$, or a velocity parameterization.

Sampling requires executing all reverse steps:

$$
x_T \to x_{T-1} \to \dots \to x_0
$$

This iterative nature makes inference **computationally expensive**.

---

## Motivation: Truncated Diffusion

The key observation is that diffusion models only guarantee that **$x_T$ is Gaussian**, not that intermediate states $x_t$ are easy to sample from.

To accelerate inference, we truncate the diffusion chain:

$$
T_\text{trunc} < T
$$

and perform sampling as:

$$
x_{T_\text{trunc}} \to x_{T_\text{trunc}-1} \to \dots \to x_0
$$

This reduces inference time by a factor of $T / T_\text{trunc}$.

### Challenge

Unlike $x_T$, the distribution of $x_{T_\text{trunc}}$:

$$
q(x_{T_\text{trunc}})
$$

is **not Gaussian** and generally unknown. Therefore, we need a method to efficiently sample from this distribution.

---

## Existing Approaches

Several strategies can be used to approximate or sample from $q(x_{T_\text{trunc}})$:

### 1. Latent Variable Models (VAE / GAN)

* Train a VAE or GAN to model samples of $x_{T_\text{trunc}}$
* During inference, sample from the latent space and decode to obtain $x_{T_\text{trunc}}$
* Pros: flexible and expressive
* Cons: additional model, unstable training (GANs), reconstruction bias (VAEs)

### 2. Optimal Transport (This Work)

* Learn a **transport map** from a simple prior (e.g., Gaussian) to $q(x_{T_\text{trunc}})$
* Avoids adversarial training
* Provides a principled geometric framework

---

## Optimal Transport Formulation

### Problem Setup

Let:

* $z \sim \mathcal{N}(0, I)$ be a simple base distribution
* $x \sim q(x_{T_\text{trunc}})$ be samples obtained by forward diffusion

We aim to learn a map $f_\phi$ such that:

$$
f_\phi(z) \sim q(x_{T_\text{trunc}})
$$

This is formulated as an **Optimal Transport problem** between distributions $\mu = \mathcal{N}(0, I)$ and $\nu = q(x_{T_\text{trunc}})$.

---

### OT Objective

The optimal transport cost is defined as:

$$
\mathcal{L}*{OT}(f*\phi) = \mathbb{E}*{z \sim \mu} [c(f*\phi(z), x)]
$$

where $c(\cdot, \cdot)$ is a cost function (typically $L_2$). In practice, we approximate OT using W,

$$
\mathcal{L}_\text{SW} = \frac{1}{K} \sum_{k=1}^K \frac{1}{N} \sum_{i=1}^N \left( \mathrm{sort} \bigl(\langle x_i, \theta_k \rangle \bigl)
\mathrm{sort} \bigl(\langle \hat{x}_i, \theta_k \rangle \bigl) \right)^2.
$$

---

## Training Pipeline

1. Sample real data $x_0 \sim p_\text{data}$
2. Apply forward diffusion to obtain $x_{T_\text{trunc}}$
3. Sample noise $z \sim \mathcal{N}(0, I)$
4. Train transport network $f_\phi$ using OT loss
5. Freeze $f_\phi$ after training

---

## Results

---







