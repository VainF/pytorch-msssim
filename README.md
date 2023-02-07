# Pytorch MS-SSIM

Fast and differentiable MS-SSIM and SSIM for pytorch.

<div>
<img src="https://github.com/VainF/Images/blob/master/pytorch_msssim/lcs.png" width="25%">

#### Structural Similarity (SSIM):   
<img src="https://github.com/VainF/Images/blob/master/pytorch_msssim/ssim.png" width="50%">

#### Multi-Scale Structural Similarity (MS-SSIM):  
<img src="https://github.com/VainF/Images/blob/master/pytorch_msssim/ms-ssim.png" width="55%">
</div>

#### Why it is faster than other versions?

Gaussian kernels used in SSIM & MS-SSIM are seperable. A [separable filter](https://en.wikipedia.org/wiki/Separable_filter) in image processing can be written as product of two more simple filters. Typically a 2-dimensional convolution operation is separated into two 1-dimensional filters. This reduces the computational costs on an $N\times M$ image with a $m\times n$ filter from $\mathcal{O}(M\cdot N \cdot m \cdot n)$ down to $\mathcal{O}(M\cdot N \cdot (m+n))$. More importantly, seperated kernels are more contiguous and thus cache-friendly than 2-D kernel, which effectively accelerates the computing of SSIM/MS-SSIM. 

# Update
#### _2020.08.21_ (v0.2.1)

3D image support from [@FynnBe](https://github.com/FynnBe)!  

#### _2020.04.30_ (v0.2)

Now (v0.2), **ssim & ms-ssim can produce consistent results as tensorflow and skimage**. A benchmark (pytorch-msssim, tensorflow and skimage) can be found in the Tests section.

# Installation

```bash
pip install pytorch-msssim
```

# Usage

### 1. Basic Usage 

```python
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)  

# calculate ssim & ms-ssim for each image
ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)

# set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
ssim_loss = 1 - ssim( X, Y, data_range=255, size_average=True) # return a scalar
ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )

# reuse the gaussian kernel with SSIM & MS_SSIM. 
ssim_module = SSIM(data_range=255, size_average=True, channel=3) # channel=1 for grayscale images
ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

ssim_loss = 1 - ssim_module(X, Y)
ms_ssim_loss = 1 - ms_ssim_module(X, Y)
```
### 2. Normalized input
If you need to calculate MS-SSIM/SSIM on normalized images, please denormalize them to the range of [0, 1] or [0, 255] first.

```python
# X: (N,3,H,W) a batch of normalized images (-1 ~ 1)
# Y: (N,3,H,W)  
X = (X + 1) / 2  # [-1, 1] => [0, 1]
Y = (Y + 1) / 2  
ms_ssim_val = ms_ssim( X, Y, data_range=1, size_average=False ) #(N,)
```

### 3. Enable nonnegative_ssim

For ssim, it is recommended to set `nonnegative_ssim=True` to avoid negative results. However, this option is set to `False` by default to keep it consistent with tensorflow and skimage.

For ms-ssim, there is no nonnegative_ssim option and the ssim reponses is forced to be non-negative to avoid NaN results.


# Tests and Examples

```bash
cd tests
```
### 1. Benchmark

```bash
# requires tf2
python tests_comparisons_tf_skimage.py 

# or skimage only
# python tests_comparisons_skimage.py 
```

Outputs:

```
Downloading test image...
===================================
             Test SSIM
===================================
====> Single Image
Repeat 100 times
sigma=0.0 ssim_skimage=1.000000 (147.2605 ms), ssim_tf=1.000000 (343.4146 ms), ssim_torch=1.000000 (92.9151 ms)
sigma=10.0 ssim_skimage=0.932423 (147.5198 ms), ssim_tf=0.932661 (343.5191 ms), ssim_torch=0.932421 (95.6283 ms)
sigma=20.0 ssim_skimage=0.785744 (152.6441 ms), ssim_tf=0.785733 (343.4085 ms), ssim_torch=0.785738 (87.5639 ms)
sigma=30.0 ssim_skimage=0.636902 (145.5763 ms), ssim_tf=0.636902 (343.5312 ms), ssim_torch=0.636895 (90.4084 ms)
sigma=40.0 ssim_skimage=0.515798 (147.3798 ms), ssim_tf=0.515801 (344.8978 ms), ssim_torch=0.515791 (96.4440 ms)
sigma=50.0 ssim_skimage=0.422011 (148.2900 ms), ssim_tf=0.422007 (345.4076 ms), ssim_torch=0.422005 (86.3799 ms)
sigma=60.0 ssim_skimage=0.351139 (146.2039 ms), ssim_tf=0.351139 (343.4428 ms), ssim_torch=0.351133 (93.3445 ms)
sigma=70.0 ssim_skimage=0.296336 (145.5341 ms), ssim_tf=0.296337 (345.2255 ms), ssim_torch=0.296331 (92.6771 ms)
sigma=80.0 ssim_skimage=0.253328 (147.6655 ms), ssim_tf=0.253328 (343.1386 ms), ssim_torch=0.253324 (82.5985 ms)
sigma=90.0 ssim_skimage=0.219404 (142.6025 ms), ssim_tf=0.219405 (345.8275 ms), ssim_torch=0.219400 (100.9946 ms)
sigma=100.0 ssim_skimage=0.192681 (144.5597 ms), ssim_tf=0.192682 (346.5489 ms), ssim_torch=0.192678 (85.0229 ms)
Pass!
====> Batch
Pass!


===================================
             Test MS-SSIM
===================================
====> Single Image
Repeat 100 times
sigma=0.0 msssim_tf=1.000000 (671.5363 ms), msssim_torch=1.000000 (125.1403 ms)
sigma=10.0 msssim_tf=0.991137 (669.0296 ms), msssim_torch=0.991086 (113.4078 ms)
sigma=20.0 msssim_tf=0.967292 (670.5530 ms), msssim_torch=0.967281 (107.6428 ms)
sigma=30.0 msssim_tf=0.934875 (668.7717 ms), msssim_torch=0.934875 (111.3334 ms)
sigma=40.0 msssim_tf=0.897660 (669.0801 ms), msssim_torch=0.897658 (107.3700 ms)
sigma=50.0 msssim_tf=0.858956 (671.4629 ms), msssim_torch=0.858954 (100.9959 ms)
sigma=60.0 msssim_tf=0.820477 (670.5424 ms), msssim_torch=0.820475 (103.4489 ms)
sigma=70.0 msssim_tf=0.783511 (671.9357 ms), msssim_torch=0.783507 (113.9048 ms)
sigma=80.0 msssim_tf=0.749522 (672.3925 ms), msssim_torch=0.749518 (120.3891 ms)
sigma=90.0 msssim_tf=0.716221 (672.9066 ms), msssim_torch=0.716217 (118.3788 ms)
sigma=100.0 msssim_tf=0.684958 (675.2075 ms), msssim_torch=0.684953 (117.9481 ms)
Pass
====> Batch
Pass
```

<div>
<img src="https://github.com/VainF/Images/blob/master/pytorch_msssim/simga_0_ssim_1.0000.png"   width="20%">
<figcaption>ssim=1.0000</figcaption>
<img src="https://github.com/VainF/Images/blob/master/pytorch_msssim/simga_50_ssim_0.4225.png"  width="20%">
<figcaption>ssim=0.4225</figcaption>
<img src="https://github.com/VainF/Images/blob/master/pytorch_msssim/simga_100_ssim_0.1924.png"    width="20%">
<figcaption>ssim=0.1924</figcaption>
</div>

### 2. MS_SSIM as loss function

See ['tests/tests_loss.py'](https://github.com/VainF/pytorch-msssim/tree/master/tests/tests_loss.py) for more details about how to use ssim or ms_ssim as loss functions

### 3. AutoEncoder

See ['tests/ae_example'](https://github.com/VainF/pytorch-msssim/tree/master/tests/ae_example)

![results](https://github.com/VainF/Images/blob/master/pytorch_msssim/ae_ms_ssim.jpg)
*left: the original image, right: the reconstructed image*

# References

[https://github.com/jorge-pessoa/pytorch-msssim](https://github.com/jorge-pessoa/pytorch-msssim)  
[https://ece.uwaterloo.ca/~z70wang/research/ssim/](https://ece.uwaterloo.ca/~z70wang/research/ssim/)  
[https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf](https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf)  
[Matlab Code](https://ece.uwaterloo.ca/~z70wang/research/iwssim/)   
[ssim & ms-ssim from tensorflow](https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/ops/image_ops_impl.py#L3314-L3438) 
