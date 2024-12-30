import os
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# Global settings
ATTN_BACKEND = 'flash_attn'  # Default backend
DEBUG = False  # Add DEBUG flag here
BACKEND = 'flash_attn'  # Change default to a valid backend


def set_attention_backend(backend: Literal['xformers', 'flash_attn', 'sdpa', 'sage', 'naive']):
    """Set the global attention backend"""
    global ATTN_BACKEND, BACKEND  # Also update BACKEND when setting attention backend
    if backend not in ['xformers', 'flash_attn', 'sdpa', 'sage', 'naive']:
        raise ValueError(f"Unsupported attention backend: {backend}")
    ATTN_BACKEND = backend
    BACKEND = backend  # Keep BACKEND in sync with ATTN_BACKEND
    os.environ['ATTN_BACKEND'] = backend
    logger.info(f"[ATTENTION] Set backend to: {backend}")


def get_attention_op():
    """Get the appropriate attention implementation"""
    if ATTN_BACKEND == 'xformers':
        try:
            import xformers.ops
            return xformers.ops.memory_efficient_attention
        except ImportError:
            logger.warning("xformers not available, falling back to naive attention")
            return None
    elif ATTN_BACKEND == 'flash_attn':
        try:
            from flash_attn import flash_attn_func
            return flash_attn_func
        except ImportError:
            logger.warning("flash_attn not available, falling back to naive attention")
            return None
    elif ATTN_BACKEND == 'sage':
        try:
            from sageattention import sageattn
            return sageattn
        except ImportError:
            logger.warning("sageattention not available, falling back to naive attention")
            return None
    elif ATTN_BACKEND == 'sdpa':
        import torch
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            return torch.nn.functional.scaled_dot_product_attention

    # Fallback to naive implementation
    logger.warning("Using naive attention implementation")
    return None


# Import attention modules after defining globals
from .modules import MultiHeadAttention, RotaryPositionEmbedder

__all__ = [
    'set_attention_backend',
    'get_attention_op',
    'ATTN_BACKEND',
    'DEBUG',
    'BACKEND',
    'MultiHeadAttention',
    'RotaryPositionEmbedder'
]