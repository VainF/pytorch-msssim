import numpy as np
import urllib
import time
from PIL import Image
from skimage.measure import compare_ssim
from pytorch_msssim import ssim, ms_ssim
import torch
import os
import sys

if __name__ == '__main__':
    print("Downloading test image...")
    if not os.path.isfile("kodim10.png"):
        urllib.request.urlretrieve(
            "http://r0k.us/graphics/kodak/kodak/kodim10.png", "kodim10.png")

    img = Image.open('kodim10.png')
    img = np.array(img).astype(np.float32)

    img_batch = []
    img_noise_batch = []
    single_image_ssim = []

    print("====> Single Image")
    # params = torch.nn.Parameter( torch.ones(img.shape[2], img.shape[0], img.shape[1]), requires_grad=True ) # C, H, W
    for sigma in range(0, 10, 1):
        noise = sigma * np.random.randn(*img.shape) + 0
        img_noise = (img + noise).astype(np.float32)

        begin = time.time()
        ssim_skimage = compare_ssim(img, img_noise, win_size=11, multichannel=True,
                                    sigma=1.5, data_range=255, use_sample_covariance=False, gaussian_weights=True)
        time_skimage = time.time()-begin

        img_torch = torch.from_numpy(img).unsqueeze(
            0).permute(0, 3, 1, 2)  # 1, C, H, W
        img_noise_torch = torch.from_numpy(
            img_noise).unsqueeze(0).permute(0, 3, 1, 2)

        img_batch.append(img_torch)
        img_noise_batch.append(img_noise_torch)

        begin = time.time()
        ssim_torch = ssim(img_noise_torch, img_torch,
                          win_size=11, data_range=255)
        time_torch = time.time()-begin

        ssim_torch = ssim_torch.numpy()
        single_image_ssim.append(ssim_torch)

        print("sigma=%f compare_ssim=%f (%f ms) ssim_torch=%f (%f ms)" % (
            sigma, ssim_skimage, time_skimage*1000, ssim_torch, time_torch*1000))
        assert (np.allclose(ssim_torch, ssim_skimage, atol=5e-4))

    print("Pass")

    print("====> Batch")
    img_batch = torch.cat(img_batch, dim=0)
    img_noise_batch = torch.cat(img_noise_batch, dim=0)
    ssim_batch = ssim(img_noise_batch, img_batch, win_size=11,
                      size_average=False, data_range=255)
    assert np.allclose(ssim_batch, single_image_ssim, atol=5e-4)
    print("Pass")
