import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from skimage.measure import compare_ssim

from pytorch_msssim import ssim, ms_ssim
import torch

X = torch.rand(1,1,190,190)
Y = 1 - X

# return negative ssim
print("Negative ssim:")
print( "skimage.measure.compare_ssim: ", compare_ssim(X.squeeze(0).permute(1,2,0).numpy(), Y.squeeze(0).permute(1,2,0).numpy(), win_size=11, multichannel=True,
                                    sigma=1.5, data_range=1, use_sample_covariance=False, gaussian_weights=True) )
print("pytorch_msssim.ssim: ", ssim( X, Y, data_range=1, size_average=False, K=(0.01, 0.03)).item() )
print("pytorch_msssim.ms_ssim: ",ms_ssim( X, Y, data_range=1, size_average=False, K=(0.01, 0.03)).item() )


# use a larger K2
print("\nLarger K2:")
print("pytorch_msssim.ssim (K2=0.4): ", ssim( X, Y, data_range=1, size_average=False, K=(0.01, 0.4)).item() )
print("pytorch_msssim.ms_ssim (K2=0.4): ",ms_ssim( X, Y, data_range=1, size_average=False, K=(0.01, 0.4)).item() )