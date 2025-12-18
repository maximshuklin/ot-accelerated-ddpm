# Diffusion Model Acceleration using Optimal Transport

## Introduction

Diffusion models have become a go-to method for generative modeling, delivering top-tier results in image, audio, and data generation. But they come with a big downside: they’re slow. Generating a single sample often means running hundreds or even thousands of denoising steps.

In this project, we looked at whether Optimal Transport could help us cut down the number of steps needed while keeping sample quality high.

---

## Experimental Configuration and Truncation Choice

We ran all experiments using the following setup:

- **Maximum diffusion steps:** 1000  
- **Truncation step:** 10  
- **Batch size:** 128  
- **Image resolution:** 32 by 32  
- **Image channels:** 3  
- **Latent dimension:** 1024  
- **Prior training epochs:** 32  

### Why We Chose Truncation Step = 10

Picking the right truncation step was tricky. We wanted to speed up inference without making the problem too hard for the generator.

If the truncation step is too small, the distribution we’re trying to model becomes really complex—almost too complex for a single-shot generator to learn reliably. On the other hand, if we set it too high, we lose most of the speedup benefit.

After some trial and error, we landed on a truncation step of 10. At this point, the truncated diffusion distribution still has enough structure to be learnable, but we also get a nice speed boost—roughly 100 times fewer diffusion steps. Also, at 10 steps, the generated images are still recognizable, unlike when we tried values above 50, where quality started to drop noticeably.

All experiments were done on CIFAR-10. To keep things fair, every model used the same generator architecture during training and evaluation.

---

## Background: Diffusion Models

### Forward (Diffusion) Process

In a diffusion model, we slowly add noise to data until it turns into something close to pure Gaussian noise. Starting from a data sample x₀ drawn from the real data distribution, the forward process is:

q(x_t | x_{t−1}) = N(√(1−β_t) x_{t−1}, β_t I),   for t = 1 up to T

This lets us jump straight to any timestep in closed form:

q(x_t | x₀) = N(√(ᾱ_t) x₀, (1−ᾱ_t) I)

where ᾱ_t is the product of all (1 − β_s) up to step t. By the final step T, q(x_T) is basically just standard Gaussian noise.

---

### Reverse (Denoising) Process

The model learns to reverse this noising process step by step—going from noise back to data. Sampling requires running through all of these reverse steps, which is what makes diffusion models so expensive at inference time.

---

## Why Truncated Diffusion?

To speed things up, we stop the diffusion process early—well before the final step T. Instead of starting from pure noise at step T, we start from a partially noisy sample at step Truncation step, where Truncation step is much smaller than T. Then we just run:

x_{Truncation step} -> x_{Truncation step−1} -> … -> x₀

This cuts inference time by roughly T / Truncation step.

But there’s a catch: unlike the fully diffused x_T, the distribution at the truncation step isn’t Gaussian anymore. We have to learn it somehow.

---

## How Others Have Tried to Solve This

### Latent Variable Models (VAE / GAN)

One common approach is to train a separate generative model—like a VAE or a GAN—on samples taken at the truncation step. Then, at inference time, you sample from that model instead of from a Gaussian.

- Good: flexible, expressive
- Bad: GANs can be unstable, VAEs often introduce blur or bias

### Optimal Transport (Our Approach)

We tried something different: using Optimal Transport to learn a generator that maps from a simple Gaussian prior directly to the truncated diffusion distribution.

The idea is to minimize a transport-based distance between the two, without adversarial training. We’re matching distributions directly, not just maximizing likelihood.

---

## Optimal Transport Setup

Let z be a random vector from a standard Gaussian, and let x be a sample from the truncated diffusion distribution at step Truncation step.

We learn a generator f_φ so that f_φ(z) looks as close as possible to the real truncated samples.

### Sliced Wasserstein Objective

Full Optimal Transport is too heavy in high dimensions, so we used the Sliced Wasserstein Distance instead. It’s simpler and works well in practice:

L_SWD =
(1/K) Σ_k (1/N) Σ_i ( sort(<x_i, θ_k>) − sort(<x̂_i, θ_k>) )²

Here, θ_k are random projection directions, and x̂_i = f_φ(z_i).

---

## How We Trained Everything

1. First, we generated a dataset of truncated diffusion samples (at step Truncation step).
2. We trained three different priors on the same data: a VAE, a GAN, and our OT-based generator.
3. All models used the same DCGAN-style generator architecture for fairness.
4. Once trained, we froze the priors.
5. Finally, we used each prior to initialize the truncated reverse diffusion process and compared results.

---

## Results

### Numbers

| Method  | FID ↓ | RecallDist ↓ |
|---------|-------|--------------|
| Vanilla | 60.10 | 18.834       |
| VAE     | 2.91  | 16.393       |
| GAN     | 0.33  | 14.004       |
| OT      | **0.75**  | **16.031**       |

Lower is better for both metrics.

Our OT-based prior did much better than starting from a plain Gaussian (vanilla). It also came close to the GAN in FID while avoiding the instability of adversarial training.

---

## Looking at the Feature Space

We also ran PCA on Inception-v3 features extracted from real and generated images, then projected down to 2D for visualization. This gives us a qualitative sense of how well the generated samples match the real distribution in a perceptually relevant space. From the plots, we noticed that the GAN prior tends to overfit—its samples cluster tightly, while the OT prior spreads out more like the real data.