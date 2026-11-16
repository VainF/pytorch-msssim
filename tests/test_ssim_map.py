import pytest
import torch

from pytorch_msssim import SSIM, ssim


@pytest.mark.parametrize("shape", [(2, 3, 13, 15), (2, 3, 7, 13, 15)])
def test_local_scores_reduce_to_ssim(shape):
    torch.manual_seed(7)
    x = torch.rand(shape, dtype=torch.double, requires_grad=True)
    y = torch.rand_like(x)
    metric = SSIM(data_range=1, win_size=5, channel=3, spatial_dims=len(shape) - 2)
    scores = metric(x, y, return_map=True)
    assert scores.shape == (*shape[:2], *(size - 4 for size in shape[2:]))
    avg = ssim(x, y, data_range=1, win_size=5, size_average=False)
    torch.testing.assert_close(scores.flatten(1).mean(1), avg)
    torch.testing.assert_close(torch.autograd.grad(scores.mean(), x)[0],
                               torch.autograd.grad(avg.mean(), x)[0])


def test_nonnegative_ssim_clamps_each_window():
    x = torch.arange(13).float().view(1, 1, 1, 13).expand(1, 1, 13, 13) / 12
    y = 1 - x
    scores = ssim(x, y, data_range=1, win_size=5, return_map=True)
    assert (scores < 0).any()
    clipped = ssim(x, y, data_range=1, win_size=5, return_map=True, nonnegative_ssim=True)
    torch.testing.assert_close(clipped, scores.relu())
