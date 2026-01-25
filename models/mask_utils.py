import torch
from torch.nn.attention.flex_attention import BlockMask, and_masks, create_block_mask


def get_block_mask(idx: torch.Tensor, eos_token_id: int, window_size: int) -> BlockMask:
    B, T = idx.shape
    doc = (idx == eos_token_id).cumsum(dim=1)

    def document(b, h, q_idx, k_idx):
        return doc[b, q_idx] == doc[b, k_idx]

    def casual_mask(b, h, q_idx, k_idx):
        return q_idx >= k_idx

    def sliding_window_mask(b, h, q_idx, k_idx):
        return q_idx - k_idx >= window_size

    return create_block_mask(
        and_masks(document, casual_mask, sliding_window_mask),
        B,
        None,
        T,
        T,
        compile=True,
    )
