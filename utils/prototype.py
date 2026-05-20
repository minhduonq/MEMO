import torch
import torch.nn.functional as F


def prototype_regularization_loss(z_new: torch.Tensor, y: torch.Tensor, prototypes: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    device = z_new.device
    classes = y.unique()

    losses = []
    for c in classes:
        c_val = c.item()
        if c_val >= prototypes.shape[0]:
            continue

        mask = (y == c)
        if mask.sum() == 0:
            continue

        mu_new = z_new[mask].mean(dim=0)            
        mu_old = prototypes[c_val].to(device)        
        losses.append((mu_new - mu_old).pow(2).sum())

    if len(losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(losses).mean()
