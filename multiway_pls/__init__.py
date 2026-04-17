"""Multi-way PCA and PLS analysis package."""

from .multiway_pca import MultiwayPCA
from .multiway_pls import MultiwayPLS
from .utils import unfold, preprocess, reverse_preprocess, variance_explained

__all__ = [
    'MultiwayPCA',
    'MultiwayPLS',
    'unfold',
    'preprocess',
    'reverse_preprocess',
    'variance_explained',
]
