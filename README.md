# Pytorch MS-SSIM

Fast and differentiable MS-SSIM and SSIM for pytorch 1.0+

All calculations will be on the same device as inputs.

# update
_2019.6.17_  
Now it is faster than compare_ssim thanks to [One-sixth's contribution](https://github.com/VainF/pytorch-msssim/issues/3)

_2019.8.15_  
[Apply to 5D tensor #6](https://github.com/VainF/pytorch-msssim/issues/6)
# Install

```bash
pip install pytorch-msssim
```
or
```bash
python setup.py install
```

# Example

```python
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# X: (N,3,H,W) a batch of RGB images (0~255)
# Y: (N,3,H,W)  
ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)

# or set 'size_average=True' to get a scalar value as loss.
ssim_loss = 1 - ssim( X, Y, data_range=255, size_average=True) # return a scalar
ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )

# or reuse windows with SSIM & MS_SSIM. 
ssim_module = SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3)
ms_ssim_module = MS_SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3)

ssim_loss = 1 - ssim_module(X, Y)
ms_ssim_loss = 1 - ms_ssim_module(X, Y)
```
**Please note that you should maximize ssim to get high-quality images. The loss function should be 1-ssim.**  
See *tests/ae_example* for more details.

# Tests

Compared with [skimage.measure.compare_ssim](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim) on CPU.

The outputs:
```
Downloading test image...
====> Single Image
sigma=0.000000 compare_ssim=1.000000 (417.248964 ms) ssim_torch=1.000000 (257.593870 ms)
sigma=1.000000 compare_ssim=0.991320 (326.905012 ms) ssim_torch=0.991320 (135.488033 ms)
sigma=2.000000 compare_ssim=0.966521 (485.862017 ms) ssim_torch=0.966520 (237.199068 ms)
sigma=3.000000 compare_ssim=0.928799 (323.492050 ms) ssim_torch=0.928797 (148.905993 ms)
sigma=4.000000 compare_ssim=0.882271 (290.801048 ms) ssim_torch=0.882267 (146.914005 ms)
sigma=5.000000 compare_ssim=0.831310 (282.787085 ms) ssim_torch=0.831306 (148.653984 ms)
sigma=6.000000 compare_ssim=0.778222 (308.619022 ms) ssim_torch=0.778217 (147.915840 ms)
sigma=7.000000 compare_ssim=0.726444 (290.637970 ms) ssim_torch=0.726438 (133.754253 ms)
sigma=8.000000 compare_ssim=0.676345 (294.582129 ms) ssim_torch=0.676339 (144.154072 ms)
sigma=9.000000 compare_ssim=0.629922 (300.610065 ms) ssim_torch=0.629916 (141.150951 ms)
Pass
====> Batch
Pass
```

# An autoencoder trained with MS_SSIM

![results](https://github.com/VainF/Images/blob/master/pytorch_msssim/ae_ms_ssim.jpg)
*left: original image, right: reconstructed image*

# References

[https://github.com/jorge-pessoa/pytorch-msssim](https://github.com/jorge-pessoa/pytorch-msssim)  
[https://ece.uwaterloo.ca/~z70wang/research/ssim/](https://ece.uwaterloo.ca/~z70wang/research/ssim/)  
[https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf](https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf)  
[Matlab Code](https://ece.uwaterloo.ca/~z70wang/research/iwssim/)  
