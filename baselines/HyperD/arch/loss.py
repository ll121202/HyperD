import torch
import torch.nn.functional as F


def dual_view_loss(y, periodic, residual, F_low):
    y_fft = torch.fft.rfft(y, dim=1)

    low_fft = torch.zeros_like(y_fft)
    low_fft[:, :F_low] = y_fft[:, :F_low]

    high_fft = torch.zeros_like(y_fft)
    high_fft[:, F_low:] = y_fft[:, F_low:]

    low_time = torch.fft.irfft(low_fft, n=y.size(1), dim=1)
    high_time = torch.fft.irfft(high_fft, n=y.size(1), dim=1)

    loss_low = F.mse_loss(periodic, low_time)
    loss_high = F.mse_loss(residual, high_time)

    return loss_low, loss_high


