"""S3-DiT 单流 diffusion transformer 模块。

第 4 步范围：可配置的最小 faithful 单流 backbone（不实现训练）。
"""

from .config import S3DiTConfig
from .s3dit import S3DiT

__all__ = ["S3DiTConfig", "S3DiT"]
