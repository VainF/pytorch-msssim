import numpy as np
import urllib
import time
from PIL import Image
from skimage.metrics import structural_similarity
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pytorch_msssim import ssim, ms_ssim
import torch


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
    N_repeat = 100
    print("====> Single Image")
    print("Repeat %d times"%(N_repeat))
    # params = torch.nn.Parameter( torch.ones(img.shape[2], img.shape[0], img.shape[1]), requires_grad=True ) # C, H, W
    for sigma in range(0, 101, 10):
        noise = sigma * np.random.rand(*img.shape)
        img_noise = (img + noise).astype(np.float32).clip(0,255)
        
        begin = time.time()
        for _ in range(N_repeat):
            ssim_skimage = structural_similarity(img, img_noise, win_size=11, multichannel=True,
                                    sigma=1.5, data_range=255, channel_axis=2, use_sample_covariance=False, gaussian_weights=True)
        time_skimage = (time.time()-begin) / N_repeat

        img_torch = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)  # 1, C, H, W
        img_noise_torch = torch.from_numpy(img_noise).unsqueeze(0).permute(0, 3, 1, 2)

        img_batch.append(img_torch)
        img_noise_batch.append(img_noise_torch)

        begin = time.time()
        for _ in range(N_repeat):
            ssim_torch = ssim(img_noise_torch, img_torch, win_size=11, data_range=255)
        time_torch = (time.time()-begin) / N_repeat

        ssim_torch = ssim_torch.numpy()
        single_image_ssim.append(ssim_torch)

        print("sigma=%f ssim_skimage=%f (%f ms) ssim_torch=%f (%f ms)" % (
            sigma, ssim_skimage, time_skimage*1000, ssim_torch, time_torch*1000))

        #Image.fromarray( img_noise.astype('uint8') ).save('simga_%d_ssim_%.4f.png'%(sigma, ssim_torch.item()))
        assert (np.allclose(ssim_torch, ssim_skimage, atol=5e-4))

    print("Pass")

    print("====> Batch")
    img_batch = torch.cat(img_batch, dim=0)
    img_noise_batch = torch.cat(img_noise_batch, dim=0)
    ssim_batch = ssim(img_noise_batch, img_batch, win_size=11,
                      size_average=False, data_range=255)
    assert np.allclose(ssim_batch, single_image_ssim, atol=5e-4)
    print("Pass")



