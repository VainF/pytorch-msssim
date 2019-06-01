# Pytorch MS-SSIM

Fast and differentiable MS-SSIM and SSIM for pytorch 1.0+

For faster calculation speed, the 2D convolution (Gaussian Blur) is replaced by two 1D convolutions.  
see [Gaussian_blur wiki](https://en.wikipedia.org/wiki/Gaussian_blur#Implementation).

All calculations will be on the same device as inputs.

# Install
```bash
python setup.py install
```
or
```bash
pip install pytorch-msssim
```

# Example

```python
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# X: (N,3,H,W) a batch of RGB images with values ranging from 0 to 255.
# Y: (N,3,H,W)  
ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)

# or set 'size_average=True' to get a scalar value as loss.
ssim_loss = ssim( X, Y, data_range=255, size_average=True) # return a scalar value
ms_ssim_loss = ms_ssim( X, Y, data_range=255, size_average=True )

# or reuse windows with SSIM & MS_SSIM. 
ssim_module = SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3)
ms_ssim_module = MS_SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3)

ssim_loss = ssim_module(X, Y)
ms_ssim_loss = ms_ssim_module(X, Y)
```

# Tests

Compared with [skimage.measure.compare_ssim](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim) on CPU.

The outputs:
```
Downloading the test image...
====> Single Image
sigma=0.000000 compare_ssim=1.000000 (291.220903 ms) ssim_torch=1.000000 (389.045000 ms)
sigma=1.000000 compare_ssim=0.991319 (302.870035 ms) ssim_torch=0.991312 (463.139057 ms)
sigma=2.000000 compare_ssim=0.966552 (416.693926 ms) ssim_torch=0.966527 (445.262909 ms)
sigma=3.000000 compare_ssim=0.928726 (305.456877 ms) ssim_torch=0.928674 (459.895134 ms)
sigma=4.000000 compare_ssim=0.882462 (303.186893 ms) ssim_torch=0.882380 (354.626179 ms)
sigma=5.000000 compare_ssim=0.831174 (279.859304 ms) ssim_torch=0.831065 (354.197025 ms)
sigma=6.000000 compare_ssim=0.778095 (295.956135 ms) ssim_torch=0.777961 (353.795052 ms)
sigma=7.000000 compare_ssim=0.726729 (304.435015 ms) ssim_torch=0.726576 (354.927063 ms)
sigma=8.000000 compare_ssim=0.677140 (287.097931 ms) ssim_torch=0.676973 (359.275103 ms)
sigma=9.000000 compare_ssim=0.630489 (282.092094 ms) ssim_torch=0.630312 (376.378059 ms)
Pass
====> Batch
Pass
```

# An autoencoder trained with MS_SSIM

![results](https://github.com/VainF/Images/blob/master/pytorch_msssim/ae_ms_ssim.jpg)
*left: original image, right: reconstructed image*

# Reference

[https://github.com/jorge-pessoa/pytorch-msssim](https://github.com/jorge-pessoa/pytorch-msssim)  
[https://ece.uwaterloo.ca/~z70wang/research/ssim/](https://ece.uwaterloo.ca/~z70wang/research/ssim/)  
[https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf](https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf)  
[Matlab Code](https://ece.uwaterloo.ca/~z70wang/research/iwssim/)  
