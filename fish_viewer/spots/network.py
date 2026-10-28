"""Small heatmap U-Net."""
import torch
from torch import nn


class CenterUNet(nn.Module):
    def __init__(self, filters=12):
        super().__init__()
        def block(inputs, outputs):
            return nn.Sequential(nn.Conv3d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU(),
                                 nn.Conv3d(outputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU())
        self.enc1 = block(3, filters)
        self.enc2 = block(filters, filters * 2)
        self.bottom = block(filters * 2, filters * 4)
        self.dec2 = block(filters * 6, filters * 2)
        self.dec1 = block(filters * 3, filters)
        self.head = nn.Conv3d(filters, 1, 1)
        nn.init.constant_(self.head.bias, -4.)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(nn.functional.max_pool3d(a, 2))
        c = self.bottom(nn.functional.max_pool3d(b, 2))
        d = self.dec2(torch.cat([nn.functional.interpolate(c, size=b.shape[-3:], mode='trilinear', align_corners=False), b], 1))
        e = self.dec1(torch.cat([nn.functional.interpolate(d, size=a.shape[-3:], mode='trilinear', align_corners=False), a], 1))
        return dict(heatmaps=(self.head(e),))


def create(engine, config):
    if engine != 'signal-center-unet3d':
        raise ValueError('Choose the signal-center U-Net engine.')
    return CenterUNet(config.filters)


def masked_heatmap_loss(logits, target, valid, positive_weight):
    weight = valid * (1 + positive_weight * target)
    loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none')
    return (loss * weight).sum() / weight.sum().clamp_min(1)


def loss(output, target, valid, config):
    heat = sum(masked_heatmap_loss(logit, nn.functional.interpolate(target, size=logit.shape[-3:], mode='trilinear', align_corners=False),
        nn.functional.interpolate(valid, size=logit.shape[-3:], mode='nearest'), config.positive_weight) / 4 ** level
        for level, logit in enumerate(output['heatmaps']))
    auxiliary = heat.new_zeros(())
    return heat + auxiliary, heat.detach(), auxiliary.detach()
