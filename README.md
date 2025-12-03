# Latent-Space Dataset Distillation with Reinforcement Learning

This project explores **dataset distillation in latent space** combined with **reinforcement learning (RL)**.  
The aim is to study whether **RL can help shape synthetic prototypes** in a latent feature space to match or beat strong baselines such as:

- Full training set in latent space (upper bound)
- Random latent subset
- k-means centroids in latent space
- Latent prototypes trained with a fixed loss schedule (no RL)


---

## High-Level Idea

1. Train a **convolutional autoencoder with a classifier head** on CIFAR-10.
   - Encoder → latent representation `z`
   - Decoder → reconstructs the image from `z`
   - Classifier head → predicts label from `z`

2. Perform all dataset distillation **in latent space**:
   - Learn **synthetic latent prototypes** per class.
   - Train a **linear classifier** on these prototypes and evaluate on real test latents.

3. Add **Reinforcement Learning** in two places:
   - **RL-Selection**: RL chooses which real latent points to keep under a per-class budget.
   - **RL-Shaping of prototypes (Actor–Critic)**: RL controls the loss weighting schedule during prototype training, starting from a strong baseline (fine-tuning).

4. Decode prototypes back to image space for **visualization & intuition**.

This gives you a full experimental pipeline for **latent + RL + dataset distillation (DD)**.

---
