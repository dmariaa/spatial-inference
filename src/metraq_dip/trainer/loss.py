import torch


def get_losses(y: torch.Tensor, y_hat: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    assert y.shape == y_hat.shape, "y and y_hat must have the same shape"

    diff = (y_hat - y) * mask
    count = mask.to(torch.bool).expand_as(diff).float().sum()

    losses = {
        'L1Loss': diff.abs().sum() / count,
        'MSELoss': diff.pow(2).sum() / count,
    }

    return losses

def get_metrics(y: torch.Tensor, y_hat: torch.Tensor, mask: torch.Tensor, pollutants: dict[int, str]):
    assert y.shape == y_hat.shape, "y and y_hat must have the same shape"

    result = []

    for i, pollutant in enumerate(pollutants):
        diff = (y_hat[:, i:i+1, ...] - y[:, i:i+1, ...]) * mask[:, i:i+1, ...].float()
        count = mask.to(torch.bool).expand_as(diff).float().sum()

        losses = {
            'L1Loss': torch.abs(diff).sum() / count,
            'MSELoss': torch.square(diff).sum() / count,
        }

        for loss in losses:
            result.append({
                'pollutant': pollutant,
                'criterion': loss,
                'loss': losses[loss].item()
            })

    return result

