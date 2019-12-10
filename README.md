# Pytorch MS-SSIM

Fast and differentiable MS-SSIM and SSIM for pytorch 1.0+

# Install

```bash
pip install pytorch-msssim
```
or
```bash
python setup.py install
```

# Updates

### _2019.12.10_  
Negative or NaN results: [#11](https://github.com/VainF/pytorch-msssim/issues/11) and [#7](https://github.com/VainF/pytorch-msssim/issues/7)

The negative results or NaN results are caused by negative covariances of input images, **which can be avoided by using a larger K2 constant (e.g. 0.4)**. See 'tests/tests_negative_ssim.py' for more details.

```python
# set K2=0.4 for more stable results
ssim( X, Y, data_range=1, size_average=False, K=(0.01, 0.4))
```

### _2019.8.15_  
Apply to 5D tensor: [#6](https://github.com/VainF/pytorch-msssim/issues/6)


### _2019.6.17_  
Now it is faster than compare_ssim thanks to [One-sixth's contribution](https://github.com/VainF/pytorch-msssim/issues/3)


# Usages

Calculations will be on the same device as input images.

```python
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)  

ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)

# set 'size_average=True' to get a scalar value as loss.
ssim_loss = 1 - ssim( X, Y, data_range=255, size_average=True) # return a scalar
ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )

# reuse windows with SSIM & MS_SSIM. 
ssim_module = SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3)
ms_ssim_module = MS_SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3)

ssim_loss = 1 - ssim_module(X, Y)
ms_ssim_loss = 1 - ms_ssim_module(X, Y)
```

# Tests

```bash
cd tests
```
### 1. Compared with [skimage.measure.compare_ssim](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim) on CPU.

```bash
python tests.py
```

<div>
<img src="https://github.com/VainF/Images/blob/master/pytorch_msssim/simga_0_ssim_1.0000.png"   width="20%">
<figcaption>ssim=1.0000</figcaption>
<img src="https://github.com/VainF/Images/blob/master/pytorch_msssim/simga_50_ssim_0.4225.png"  width="20%">
<figcaption>ssim=0.4225</figcaption>
<img src="https://github.com/VainF/Images/blob/master/pytorch_msssim/simga_100_ssim_0.1924.png"    width="20%">
<figcaption>ssim=0.1924</figcaption>
</div>

The outputs:
```
Downloading test image...
====> Single Image
sigma=0.000000 compare_ssim=1.000000 (275.226831 ms) ssim_torch=1.000000 (462.517738 ms)
sigma=10.000000 compare_ssim=0.932497 (389.491558 ms) ssim_torch=0.932494 (63.863516 ms)
sigma=20.000000 compare_ssim=0.785664 (266.695976 ms) ssim_torch=0.785658 (46.617031 ms)
sigma=30.000000 compare_ssim=0.637369 (275.762081 ms) ssim_torch=0.637362 (55.842876 ms)
sigma=40.000000 compare_ssim=0.515707 (236.553907 ms) ssim_torch=0.515700 (45.801163 ms)
sigma=50.000000 compare_ssim=0.422497 (264.705896 ms) ssim_torch=0.422491 (46.895742 ms)
sigma=60.000000 compare_ssim=0.350707 (234.748363 ms) ssim_torch=0.350702 (44.762611 ms)
sigma=70.000000 compare_ssim=0.295998 (210.025072 ms) ssim_torch=0.295993 (45.758247 ms)
sigma=80.000000 compare_ssim=0.253552 (250.259876 ms) ssim_torch=0.253547 (96.461058 ms)
sigma=90.000000 compare_ssim=0.219344 (263.813257 ms) ssim_torch=0.219340 (49.159765 ms)
sigma=100.000000 compare_ssim=0.192421 (258.941889 ms) ssim_torch=0.192418 (47.627449 ms)
Pass
====> Batch
Pass
```



### 2. Avoid negative or NaN results
```bash
python tests_negative_ssim.py
```

The outputs:
```
Negative ssim:
skimage.measure.compare_ssim:  -0.96587564223943
pytorch_msssim.ssim:  -0.9658759832382202
pytorch_msssim.ms_ssim:  nan

Larger K2:
pytorch_msssim.ssim (K2=0.4):  0.0062743788585066795
pytorch_msssim.ms_ssim (K2=0.4):  0.6563504934310913
```

### 3. train your autoencoder with MS_SSIM

see 'tests/ae_example'

![results](https://github.com/VainF/Images/blob/master/pytorch_msssim/ae_ms_ssim.jpg)
*left: the original image, right: the reconstructed image*

# References

[https://github.com/jorge-pessoa/pytorch-msssim](https://github.com/jorge-pessoa/pytorch-msssim)  
[https://ece.uwaterloo.ca/~z70wang/research/ssim/](https://ece.uwaterloo.ca/~z70wang/research/ssim/)  
[https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf](https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf)  
[Matlab Code](https://ece.uwaterloo.ca/~z70wang/research/iwssim/)  
