import torch
import torch.nn.functional as F
def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp( -(coords**2) / (2*sigma**2) )
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0) 

def gaussian_filter(input, win):
    N,C,H,W = input.shape
    out = F.conv2d( input, win, stride=1, padding=0, groups=C)
    out = out.transpose(2,3).contiguous()
    out = F.conv2d( out, win, stride=1, padding=0, groups=C)
    return out.transpose(2,3).contiguous()

def _ssim(X, Y, win, data_range=255, size_average=True, full=False):
    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation=1.0

    # SSIM
    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2
    
    #####################################
    # concat and 1d conv for faster conv
   
    concat_input = torch.cat( [X, Y, X*X, Y*Y, X*Y], dim=1) 
    concat_win = win.repeat(5,1,1,1).to(X.device, dtype=X.dtype) #win.repeat(5, 1, 1, 1)
    concat_out = gaussian_filter( concat_input, concat_win  )
    # unpack from conv output
    mu1, mu2, sigma1_sq, sigma2_sq, sigma12 = ( concat_out[:, idx*channel:(idx+1)*channel,:,: ] for idx in range(5) )

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * ( sigma1_sq - mu1_sq )
    sigma2_sq = compensation * ( sigma2_sq - mu2_sq )
    sigma12 = compensation * ( sigma12 - mu1_mu2  )

    ##########################
    # implementation of original repo

    #_mu1 = F.conv2d( X, win, stride=1, padding=0, groups=channel)
    #_mu2 = F.conv2d( Y, win, stride=1, padding=0, groups=channel)

    #mu1_sq = mu1.pow(2)
    #mu2_sq = mu2.pow(2)
    #mu1_mu2 = mu1 * mu2

    #sigma1_sq = compensation * ( F.conv2d( X*X, win, stride=1, padding=0, groups=channel) - mu1_sq )
    #sigma2_sq = compensation * ( F.conv2d( Y*Y, win, stride=1, padding=0, groups=channel) - mu2_sq )
    #sigma12 = compensation * ( F.conv2d( X*Y, win, stride=1, padding=0, groups=channel) - mu1_mu2 )

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ( (2 * mu1_mu2 + C1)  / (mu1_sq + mu2_sq + C1)  ) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1) # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)
    
    if full:
        return ssim_val, cs
    else:
        return ssim_val

def ssim(X, Y, win_size=11, win=None, data_range=255, size_average=True, full=False):
    """ SSIM
    """
    if len(X.shape)!=4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type()==Y.type():
        raise ValueError('Input images must have the same dtype.')
    
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')
    
    win_sigma = 1.5
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim( X, Y,
                            win=win,
                            data_range=data_range,
                            size_average=False,
                            full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()
    
    if full:
        return ssim_val, cs
    else:
        return ssim_val

def ms_ssim(X, Y, win_size=11, win=None, data_range=255, size_average=True, full=False, weights=None):
    if weights is None:
        weights = torch.FloatTensor( [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] ).to(X.device, dtype=X.dtype)
    
    win_sigma = 1.5
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]
    
    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        # Only last ssim value will be used
        ssim_val, cs = _ssim( X, Y,
                                win=win,
                                data_range=data_range,
                                size_average=False,
                                full=True)
        mcs.append(cs)
        
        padding = ( X.shape[2]%2, X.shape[3]%2 )
        X = F.avg_pool2d( X, kernel_size=2, padding=padding )
        Y = F.avg_pool2d( Y, kernel_size=2, padding=padding )

    mcs = torch.stack(mcs, dim=0) # mcs, (level, batch)
                                  # weights, (level)
    msssim_val = torch.prod( (mcs[:-1] ** weights[:-1].unsqueeze(1)) * (ssim_val ** weights[-1]), dim=0) # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, sigma=1.5, size_average=True, data_range=None, channel=3):
        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(win_size, sigma).repeat(channel,1,1,1)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        return ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average)

class MS_SSIM(torch.nn.Module):
    def __init__(self, win_size=11, sigma=1.5, size_average=True, data_range=None, channel=3):
        super(MS_SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(win_size, sigma).repeat(channel,1,1,1)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        return ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range)
