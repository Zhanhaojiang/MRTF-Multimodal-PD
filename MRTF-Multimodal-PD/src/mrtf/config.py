from dataclasses import dataclass

@dataclass
class Config:
    d: int = 512
    voice_freq_bins: int = 128
    voice_time_steps: int = 256
    mri_size: int = 224
    sensor_channels: int = 6
    sensor_len: int = 200

    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    grad_clip: float = 1.0

    lambda_rl: float = 0.5
    lambda_reg: float = 1e-4

    alpha1_acc: float = 1.0
    alpha2_ecs: float = 0.5
    alpha3_err: float = 1.0

    gamma: float = 0.95
