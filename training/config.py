"""
Swara Model Configuration & Hyperparameters.

Single centralized location for model architectural configurations.
Allows future comparisons (e.g. 32-channel vs 64-channel DS-CNN) without
modifying multiple files.
"""

from typing import Tuple

# Current default architecture
DEFAULT_NUM_FILTERS: int = 64
DEFAULT_NUM_CLASSES: int = 3
DEFAULT_INPUT_SHAPE: Tuple[int, int, int] = (49, 10, 1)

# Alternative exploration widths (e.g., for ultra-low RAM profiles)
CANDIDATE_CHANNEL_WIDTHS = [16, 24, 32, 48, 64]
PREFERRED_LOW_RAM_CANDIDATE: int = 32

CLASSES = ["silence", "unknown", "swara"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}
