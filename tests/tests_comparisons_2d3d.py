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


if __name__ == "__main__":
    # comparing 2d and 3d implementation. A thin 3d slice should result in a similar measure as a 2d image.
    print("Downloading test image...")
    if not os.path.isfile("kodim10.png"):
        urllib.request.urlretrieve("http://r0k.us/graphics/kodak/kodak/kodim10.png", "kodim10.png")

    img = Image.open("kodim10.png")
    img = np.array(img).astype(np.float32)

    img_batch = []
    img_noise_batch = []
    single_image_ssim = []
    img_batch_3d = []
    img_noise_batch_3d = []
    single_image_ssim_3d = []
    N_repeat = 1
    print("====> Single Image")
    print("Repeat %d times" % (N_repeat))
    # params = torch.nn.Parameter( torch.ones(img.shape[2], img.shape[0], img.shape[1]), requires_grad=True ) # C, H, W
    for sigma in range(0, 101, 10):
        noise = sigma * np.random.rand(*img.shape)
        img_noise = (img + noise).astype(np.float32).clip(0, 255)

        img_torch = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)  # 1, C, H, W
        img_noise_torch = torch.from_numpy(img_noise).unsqueeze(0).permute(0, 3, 1, 2)

        img_batch.append(img_torch)
        img_noise_batch.append(img_noise_torch)

        begin = time.perf_counter()
        for _ in range(N_repeat):
            ssim_torch = ssim(img_noise_torch, img_torch, win_size=11, data_range=255)

        time_torch = (time.perf_counter() - begin) / N_repeat

        ssim_torch = ssim_torch.numpy()
        single_image_ssim.append(ssim_torch)


        img_torch_3d = img_torch.unsqueeze(2).expand(-1, -1, 11, -1, -1)  # 1, C, H, W -> 1, C, T, H, W
        img_noise_torch_3d = img_noise_torch.unsqueeze(2).expand(-1, -1, 11, -1, -1)

        img_batch_3d.append(img_torch_3d)
        img_noise_batch_3d.append(img_noise_torch_3d)

        begin = time.perf_counter()
        for _ in range(N_repeat):
            ssim_torch_3d = ssim(img_noise_torch_3d, img_torch_3d, win_size=11, data_range=255)

        time_torch_3d = (time.perf_counter() - begin) / N_repeat

        ssim_torch_3d = ssim_torch_3d.numpy()
        single_image_ssim_3d.append(ssim_torch_3d)


        print(
            "sigma=%f ssim_torch=%f (%f ms) ssim_torch_3d=%f (%f ms)"
            % (sigma, ssim_torch, time_torch * 1000, ssim_torch_3d, time_torch_3d * 1000)
        )

        # Image.fromarray( img_noise.astype('uint8') ).save('simga_%d_ssim_%.4f.png'%(sigma, ssim_torch.item()))
        assert np.allclose(ssim_torch, ssim_torch_3d, atol=5e-4)

    print("Pass")
