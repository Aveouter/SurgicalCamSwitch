import torch
import torch.nn as nn


def divide_no_nan(a, b):
    result = a / b
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


class mape_loss(nn.Module):
    def forward(self, insample, freq, forecast, target, mask):
        weights = divide_no_nan(mask, target)
        return torch.mean(torch.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def forward(self, insample, freq, forecast, target, mask):
        denominator = torch.abs(forecast) + torch.abs(target)
        return 200 * torch.mean(divide_no_nan(torch.abs(forecast - target), denominator) * mask)


class mase_loss(nn.Module):
    def forward(self, insample, freq, forecast, target, mask):
        scale = torch.mean(torch.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        weights = divide_no_nan(mask, scale[:, None])
        return torch.mean(torch.abs(target - forecast) * weights)
