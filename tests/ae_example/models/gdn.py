import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
import math

# https://arxiv.org/pdf/1611.01704.pdf

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        ctx.save_for_backward(inputs, inputs.new_ones(1)*bound )
        return inputs.clamp(min=bound)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, bound = ctx.saved_tensors

        pass_through_1 = (inputs >= bound)
        pass_through_2 = (grad_output < 0)

        pass_through = (pass_through_1 | pass_through_2)
        return pass_through.type(grad_output.dtype) * grad_output, None
    
class GDN(nn.Module):
    def __init__(self, 
                num_features,
                inverse=False,
                gamma_init=.1,
                beta_bound=1e-6,
                gamma_bound=0.0,
                reparam_offset=2**-18,
            ):
        super(GDN, self).__init__()
        self._inverse = inverse
        self.num_features = num_features
        self.reparam_offset = reparam_offset
        self.pedestal = self.reparam_offset**2

        beta_init = torch.sqrt( torch.ones(num_features, dtype=torch.float) + self.pedestal )
        gama_init = torch.sqrt( torch.full( (num_features, num_features), fill_value=gamma_init, dtype=torch.float ) 
                                        * torch.eye(num_features, dtype=torch.float) + self.pedestal )

        self.beta = nn.Parameter( beta_init )
        self.gamma = nn.Parameter( gama_init )

        self.beta_bound = (beta_bound + self.pedestal) ** 0.5
        self.gamma_bound = (gamma_bound + self.pedestal) ** 0.5

    def _reparam(self, var, bound):
        var = LowerBound.apply( var, bound )
        return (var**2) - self.pedestal

    def forward(self, x):
        gamma = self._reparam( self.gamma, self.gamma_bound ).view(self.num_features, self.num_features, 1,1) # expand to (C, C, 1, 1)
        beta = self._reparam( self.beta, self.beta_bound )
        norm_pool = F.conv2d( x**2, gamma, bias=beta, stride=1, padding=0)
        norm_pool = torch.sqrt(norm_pool)

        if self._inverse:
            norm_pool = x * norm_pool 
        else:
            norm_pool = x / norm_pool  
        return norm_pool

        

        
