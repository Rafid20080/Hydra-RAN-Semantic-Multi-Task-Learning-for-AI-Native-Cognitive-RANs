"""
Hydra-RAN Configuration Module
IEEE TCCN Reproducibility Package
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RandomConfig:
    GLOBAL_SEED: int = 42
    MONTE_CARLO_BASE_SEED: int = 1000


@dataclass(frozen=True)
class DatasetConfig:
    INPUT_DIMENSION: int = 64
    LATENT_DIMENSION: int = 128
    NUM_CELLS: int = 7
    NUM_BEAMS: int = 64
    NUM_MODALITIES: int = 4


@dataclass(frozen=True)
class ModelConfig:
    # VIB
    LAMBDA_VIB: float = 0.1
    
    # Task-relevance gating
    LAMBDA_MINE: float = 0.01
    CRITIC_UPDATE_INTERVAL: int = 10
    
    # Temporal smoothing
    BETA: float = 0.3
    TASK_SENSITIVITY_WEIGHTING: bool = True
    
    # Semantic regularizer
    LAMBDA_1: float = 1e-4  # Sparsity
    LAMBDA_2: float = 1e-4  # Low-rank
    LAMBDA_3: float = 1e-3  # Mask consistency
    LAMBDA_4: float = 1e-3  # Physics
    LAMBDA_5: float = 1e-3  # Beam coherence


@dataclass(frozen=True)
class TrainingConfig:
    NUM_EPOCHS: int = 200
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-3
    MIN_LEARNING_RATE: float = 1e-5
    LR_DECAY: float = 0.995
    EARLY_STOPPING_PATIENCE: int = 25


@dataclass(frozen=True)
class PPOConfig:
    NUM_STEPS: int = 2000000
    LEARNING_RATE: float = 3e-4
    GAMMA: float = 0.99
    LAMBDA_GAE: float = 0.95
    CLIP_RATIO: float = 0.2
    ENTROPY_BETA: float = 0.01
    NUM_EPOCHS: int = 10
    NUM_MINIBATCHES: int = 64


class Config:
    RANDOM = RandomConfig()
    DATASET = DatasetConfig()
    MODEL = ModelConfig()
    TRAINING = TrainingConfig()
    PPO = PPOConfig()
    
    @staticmethod
    def initialize_random_seed(seed=None):
        if seed is None:
            seed = Config.RANDOM.GLOBAL_SEED
        np.random.seed(seed)
