import logging
import math

import torch
import torch.nn as nn


class EntLossClassification(nn.Module):
    r"""
        A Geometry-Aware Efficient Algorithm for Compositional Entropic Risk Minimization (Extreme Classification).

        Args:
            data_size (int): number of samples in the training dataset (default: ``100000``)
            alpha (float): the step size for SCENT (in log scale, i.e., the real step size is exp(alpha)) (default: ``10.0``)
            gamma (float): the moving average factor for SOX, in range the range of (0.0, 1.0) (default: ``0.9``)
            is_scent (bool): whether to use SCENT or SOX (default: ``True``)

        Reference:
            .. [1] Wei, X., Zhou, L., Wang, B., Lin, C.J. and Yang, T., 2026.
                   A Geometry-Aware Efficient Algorithm for Compositional Entropic Risk Minimization.
                   arXiv preprint arXiv:2602.02877.
    """
    def __init__(self,
                 data_size: int,
                 alpha: float = 10.0,
                 gamma: float = 0.9,
                 is_scent: bool = True,
                 alpha_multiplier: float = 1.0,
                 alpha_mode: str = "fixed"   # NEW
                 ) -> None:
        super().__init__()
        self.data_size = data_size
        self.alpha = alpha
        self.alpha_current_val = alpha
        self.gamma_orig = gamma
        self.gamma = gamma
        self.is_scent = is_scent
        self.nu = torch.zeros(data_size, device="cpu").reshape(-1, 1)
        self.alpha_multiplier = alpha_multiplier
        # NEW: alpha strategy selector
        self.alpha_mode = alpha_mode

    # -----------------------------
    # Alpha strategies
    # -----------------------------
    def alpha_fixed(self, epoch):
        return self.alpha

    def get_alpha_current(self, epoch):
        return self.alpha_current_val

    def alpha_cosine(self, epoch, max_epoch=20):
        alpha_start = 2.0
        alpha_end = 0.1
        return alpha_end + 0.5 * (alpha_start - alpha_end) * (
            1 + math.cos(epoch / max_epoch * math.pi)
        )

    def alpha_exponential(self, epoch):
        # Starts at 2.0 and decays by 20% every epoch
        return 2.0 * (0.8 ** epoch) + 0.1

    def alpha_random(self, epoch):
        return random.uniform(2.0, 10.0)

    def alpha_mixed(self, epoch):
        f = random.choice([
            self.alpha_fixed,
            self.alpha_cosine,
            self.alpha_random
        ])
        return f(epoch)

    def get_alpha(self, epoch):
        if self.alpha_mode == "fixed":
            return self.alpha_fixed(epoch)
        elif self.alpha_mode == "cosine":
            return self.alpha_cosine(epoch)
        elif self.alpha_mode == "random":
            return self.alpha_random(epoch)
        elif self.alpha_mode == "mixed":
            return self.alpha_mixed(epoch)
        elif self.alpha_mode == "expdecay":
            return self.alpha_exponential(epoch)
        else:
            raise ValueError(f"Unknown alpha_mode: {self.alpha_mode}")

    def adjust_gamma(self, epoch: int, max_epoch: int) -> None:
        if not self.is_scent:
            self.gamma = 0.5 * (1.0 - self.gamma_orig) * (1 + torch.cos(torch.tensor(epoch / max_epoch * math.pi))) + self.gamma_orig
            logging.info(f"Adjusted gamma to {self.gamma:.6f} at epoch {epoch}")

    def forward(self,
                logits: torch.Tensor,
                indices: torch.Tensor,
                epoch: int = 0   # NEW
                ) -> dict:
        nu = self.nu[indices].to(logits.device)

        # update nu
        # check which nu are not initialized
        uninit_idx = torch.nonzero(nu == 0.0, as_tuple=True)[0]
        exp_logits_mean = torch.sum(torch.exp(logits), dim=-1, keepdim=True).detach() / (logits.shape[1] - 1)
        if self.is_scent:
            alpha_val = self.get_alpha(epoch)
            self.alpha_current_val = alpha_val
            alpha_val = math.exp(alpha_val)  # log-scale to real scale
            nu = nu + torch.log(
                1 + alpha_val * exp_logits_mean *
                torch.exp(nu * (self.alpha_multiplier - 1.0))
            ) - torch.log(
                1 + alpha_val * torch.exp(nu * self.alpha_multiplier)
            )
        else:
            b = math.log(1 - self.gamma) + nu
            w = math.log(self.gamma) + torch.log(exp_logits_mean)
            nu = torch.max(b, w) - torch.log(torch.sigmoid(torch.abs(b - w)))
        if uninit_idx.shape[0] > 0:
            nu[uninit_idx] = torch.log(exp_logits_mean[uninit_idx])
        self.nu[indices] = nu.cpu()

        # compute loss
        loss = torch.mean(torch.sum(torch.exp(logits - nu), dim=-1, keepdim=True) / (logits.shape[1] - 1))
        loss_dict = {"loss": loss}
        return loss_dict
