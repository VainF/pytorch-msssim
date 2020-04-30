# Adapted from https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/max_ssim.py

import torch
from torch.autograd import Variable
from torch import optim
from PIL import Image
import numpy as np
import sys, os
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
npImg1 = np.array(Image.open("einstein.png"))

img1 = torch.from_numpy(npImg1).float().unsqueeze(0).unsqueeze(0)/255.0
img2 = torch.rand(img1.size())

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

img1 = Variable( img1,  requires_grad=False)
img2 = Variable( img2,  requires_grad=True)

ssim_value = ssim(img1, img2).item()
print("Initial ssim:", ssim_value)

ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)

optimizer = optim.Adam([img2], lr=0.01)

while ssim_value < 0.9999:
    optimizer.zero_grad()
    _ssim_loss = 1-ssim_loss(img1, img2)
    _ssim_loss.backward()
    optimizer.step()

    ssim_value = ssim(img1, img2).item()
    print(ssim_value)

img2_ = (img2 * 255.0).squeeze()
np_img2 = img2_.detach().cpu().numpy().astype(np.uint8)
Image.fromarray(np_img2).save('results.png')
