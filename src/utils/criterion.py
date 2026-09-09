import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CombinedVisualLossConfig:
    ssim_window_size: int = 11
    ssim_sigma: float = 1.5
    ssim_k1: float = 0.01
    ssim_k2: float = 0.03
    ssim_max_val: float = 1.0
    perceptual_layers: Tuple[str, ...] = ('relu1_2', 'relu2_2', 'relu3_4', 'relu4_4')
    perceptual_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    perceptual_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    perceptual_use_l1: bool = True
    norm_dssim_min: float = 0.0
    norm_dssim_max: float = 1.0
    norm_l2_min: float = 0.0
    norm_l2_max: float = 1.0
    norm_perceptual_min: float = 0.0
    norm_perceptual_max: float = 10.0
    init_dssim_weight: float = 1.0
    init_l2_weight: float = 1.0
    init_perceptual_weight: float = 1.0
    requires_grad: bool = True
    perceptual_weights: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Compute derived SSIM constants."""
        self.ssim_c1 = (self.ssim_k1 * self.ssim_max_val) ** 2
        self.ssim_c2 = (self.ssim_k2 * self.ssim_max_val) ** 2


class CombinedVisualLoss(nn.Module):
    def __init__(self, config: Optional[CombinedVisualLossConfig] = None):
        """
        Args:
            config: Configuration object with all hyperparameters
        """
        super(CombinedVisualLoss, self).__init__()
        self.config = config if config is not None else CombinedVisualLossConfig()
        cfg = self.config

        self.log_weights = nn.Parameter(
            torch.log(torch.tensor([
                cfg.init_dssim_weight,
                cfg.init_l2_weight,
                cfg.init_perceptual_weight
            ])),
            requires_grad=cfg.requires_grad
        )
        self._build_ssim_kernel()
        self._build_perceptual_layers()

    def _build_ssim_kernel(self):
        """Create 2D Gaussian window for SSIM computation."""
        cfg = self.config
        half = cfg.ssim_window_size // 2
        x = torch.linspace(-half, half, cfg.ssim_window_size)
        gauss = torch.exp(-x.pow(2) / (2 * cfg.ssim_sigma ** 2))
        gauss = gauss / gauss.sum()

        # 2D kernel via outer product: (window, window)
        kernel2d = gauss.unsqueeze(0) * gauss.unsqueeze(1)

        # 4D depthwise-ready: (1, 1, window, window)
        kernel = kernel2d.unsqueeze(0).unsqueeze(0)
        self.register_buffer('ssim_kernel', kernel)

    def _build_perceptual_layers(self):
        """Initialize VGG19 perceptual loss layers."""
        cfg = self.config
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        # Map layer names to feature indices (vgg19.features)
        layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_4': 17,
            'relu4_4': 26,
            'relu5_4': 35
        }

        self.perceptual_layers = nn.ModuleList()
        for layer_name in cfg.perceptual_layers:
            if layer_name in layer_map:
                idx = layer_map[layer_name]
                self.perceptual_layers.append(vgg.features[:idx])

        # Freeze perceptual layers
        for param in self.perceptual_layers.parameters():
            param.requires_grad_(False)
        self.perceptual_layers.eval()

        # VGG normalization stats
        self.register_buffer(
            'perceptual_mean',
            torch.tensor(cfg.perceptual_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'perceptual_std',
            torch.tensor(cfg.perceptual_std).view(1, 3, 1, 1)
        )
        self.perceptual_use_l1 = cfg.perceptual_use_l1

        if cfg.perceptual_weights is not None:
            self.register_buffer('per_layer_weights', cfg.perceptual_weights)
        else:
            self.register_buffer(
                'per_layer_weights',
                torch.ones(len(self.perceptual_layers))
            )

    def _ssim_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute SSIM map for each pixel."""
        cfg = self.config
        c = x.size(1)
        # Expand (1, 1, k, k) -> (c, 1, k, k) for depthwise convolution
        kernel = self.ssim_kernel.expand(c, -1, -1, -1)
        pad = cfg.ssim_window_size // 2

        mu_x = F.conv2d(x, kernel, padding=pad, groups=c)
        mu_y = F.conv2d(y, kernel, padding=pad, groups=c)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, kernel, padding=pad, groups=c) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, kernel, padding=pad, groups=c) - mu_y_sq
        sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=c) - mu_xy

        ssim_map = ((2 * mu_xy + cfg.ssim_c1) * (2 * sigma_xy + cfg.ssim_c2)) / \
                   ((mu_x_sq + mu_y_sq + cfg.ssim_c1) *
                    (sigma_x_sq + sigma_y_sq + cfg.ssim_c2))

        return ssim_map

    def _dssim_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ssim_map = self._ssim_map(x, y)
        return (1 - ssim_map).mean()

    def _l2_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, y)

    def _perceptual_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.perceptual_mean.to(x.device)) / self.perceptual_std.to(x.device)
        y_norm = (y - self.perceptual_mean.to(y.device)) / self.perceptual_std.to(y.device)

        total_loss = torch.zeros((), device=x.device)
        weights = self.per_layer_weights.to(x.device)
        for i, layer in enumerate(self.perceptual_layers):
            x_feat = layer(x_norm)
            y_feat = layer(y_norm)
            if self.perceptual_use_l1:
                layer_loss = F.l1_loss(x_feat, y_feat)
            else:
                layer_loss = F.mse_loss(x_feat, y_feat)
            total_loss += weights[i] * layer_loss

        return total_loss / weights.sum()

    def _normalize_loss(self, loss: torch.Tensor, loss_type: str) -> torch.Tensor:
        cfg = self.config
        bounds = {
            'dssim': (cfg.norm_dssim_min, cfg.norm_dssim_max),
            'l2': (cfg.norm_l2_min, cfg.norm_l2_max),
            'perceptual': (cfg.norm_perceptual_min, cfg.norm_perceptual_max)
        }
        min_val, max_val = bounds[loss_type]
        normalized = (loss - min_val) / (max_val - min_val + 1e-8)
        return torch.clamp(normalized, 0.0, 1.0)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

        dssim = self._dssim_loss(pred, gt)
        l2 = self._l2_loss(pred, gt)
        perceptual = self._perceptual_loss(pred, gt)

        dssim_norm = self._normalize_loss(dssim, 'dssim')
        l2_norm = self._normalize_loss(l2, 'l2')
        perceptual_norm = self._normalize_loss(perceptual, 'perceptual')

        weights = F.softmax(self.log_weights, dim=0)
        total = weights[0] * dssim_norm \
                + weights[1] * l2_norm \
                + weights[2] * perceptual_norm

        return {'dssim': dssim_norm,
                'l2': l2_norm,
                'perceptual': perceptual_norm,
                'total': total,
                'weights': weights.detach()}

    def get_difference_maps(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        ssim_map = self._ssim_map(pred, gt)
        dssim_map = 1 - ssim_map

        l2_map = (pred - gt).pow(2).mean(dim=1, keepdim=True)

        pred_norm = (pred - self.perceptual_mean.to(pred.device)) / self.perceptual_std.to(pred.device)
        gt_norm = (gt - self.perceptual_mean.to(gt.device)) / self.perceptual_std.to(gt.device)

        pred_features = []
        gt_features = []
        for layer in self.perceptual_layers:
            pred_features.append(layer(pred_norm))
            gt_features.append(layer(gt_norm))

        return {
            'dssim_map': dssim_map,
            'l2_map': l2_map,
            'perceptual_pred_maps': pred_features,
            'perceptual_gt_maps': gt_features
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    preds = torch.normal(0, 1, (64, 3, 224, 224))
    target = torch.normal(0, 1, (64, 3, 224, 224))

    config = CombinedVisualLossConfig()
    module = CombinedVisualLoss(config)
    losses = module(preds, target)
    for (name, value) in losses.items():
        if value is not None:
            print(name, value)