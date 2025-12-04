# config.py

SEED = 0
AE_LATENT_DIM = 128
AE_EPOCHS = 12       # can increase for better latent space

NUM_CLASSES = 10

# Distillation / prototypes
M_PER_CLASS = 10              # prototypes per class
BASE_LAMBDA_REAL = 1.0        # real-data loss weight (baseline & RL base)
BASE_LAMBDA_DIV = 0.5         # diversity loss weight (baseline & RL base)
PROTO_STEPS_BASE = 350        # steps for baseline prototypes
PROTO_STEPS_RL_EPISODE = 80   # steps per RL episode for RL fine-tuning

BATCH_SIZE_REAL_BASE = 256    # real batch size for baseline proto training
BATCH_SIZE_REAL_RL = 128      # real batch size for RL-driven proto training

# RL for selection (subset)
SEL_BUDGET_PER_CLASS = 10     # per-class budget (for random, RL-selected, k-means, etc.)
SEL_EPISODES = 40
SEL_GAMMA = 1.0
SEL_LR = 1e-4
CLASSIFIER_EPOCHS_RL_REWARD = 15

# RL for prototype shaping (Actor–Critic)
PROTO_RL_EPISODES = 60
PROTO_RL_GAMMA = 1.0
PROTO_RL_LR = 1e-4
CRITIC_WEIGHT = 0.5

# Latent pool size for experiments
MAX_TRAIN_POINTS = 20000      # subsample training latents for speed

# Classifier training (evaluation)
CLASSIFIER_EPOCHS_EVAL = 40

PRINT_EVERY_SEL = 5
PRINT_EVERY_PROTO_RL = 5
