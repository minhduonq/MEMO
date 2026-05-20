import torch
import torch.nn.functional as F
from typing import Optional

# hàm tính toán tỉ lệ khoảng cách dựa trên 3 điểm i, j, k
def relative_geometry_loss_per_class(
    z_old: torch.Tensor,          
    z_new: torch.Tensor,          
    y_old: torch.Tensor,          
    max_classes_per_step: int = 10,
    anchors_per_class: int = 2,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    device = z_old.device
    assert z_old.shape == z_new.shape, "z_old and z_new must have identical shape"
    assert y_old.dtype == torch.long, "y_old should be long tensor with class ids"

    # Nếu trong buffer có hơn 10 class (từ task 2) thì random 10 class để giảm tính toán
    all_classes = y_old.unique()
    if len(all_classes) > max_classes_per_step:
        perm = torch.randperm(len(all_classes), device=device)
        selected_classes = all_classes[perm[:max_classes_per_step]]
    else:
        selected_classes = all_classes

    losses = []

    for c in selected_classes:
        idx_c = torch.nonzero(y_old == c, as_tuple=False).squeeze(1)
        if idx_c.numel() < 2:         # cần 1 cái anchor và 1 cái positive
            continue

        idx_not_c = torch.nonzero(y_old != c, as_tuple=False).squeeze(1)
        if idx_not_c.numel() == 0:
            continue

        n_anchors = min(anchors_per_class, idx_c.numel())
        anchor_ids = idx_c[torch.randperm(idx_c.numel(), device=device)[:n_anchors]]

        for a_id in anchor_ids:
            # anchor
            z_old_a = z_old[a_id]          
            z_new_a = z_new[a_id]          

            # positive (same class, different exemplar) 
            pos_candidates = idx_c[idx_c != a_id]
            if pos_candidates.numel() == 0:
                continue
            p_id = pos_candidates[torch.randint(pos_candidates.numel(), (1,), device=device)]
            z_old_p = z_old[p_id]
            z_new_p = z_new[p_id]

            # negative (different class)
            n_id = idx_not_c[torch.randint(idx_not_c.numel(), (1,), device=device)]
            z_old_n = z_old[n_id]
            z_new_n = z_new[n_id]

            # Euclidean distances
            d_old_pos = torch.norm(z_old_a - z_old_p, p=2)
            d_old_neg = torch.norm(z_old_a - z_old_n, p=2)
            d_new_pos = torch.norm(z_new_a - z_new_p, p=2)
            d_new_neg = torch.norm(z_new_a - z_new_n, p=2)

            # Ratio
            r_old = d_old_pos / (d_old_neg + eps)
            r_new = d_new_pos / (d_new_neg + eps)

            # Triplet loss
            loss_triplet = (r_new - r_old).abs()
            losses.append(loss_triplet)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    loss_stack = torch.stack(losses)
    if reduction == "mean":
        return loss_stack.mean()
    elif reduction == "sum":
        return loss_stack.sum()
    else:  # "none"
        return loss_stack

