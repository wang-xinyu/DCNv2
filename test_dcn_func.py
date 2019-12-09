#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from dcn_v2 import DCN, DCNv2
from dcn_v2_func import DCNv2Function

deformable_groups = 1
N, inC, inH, inW = 2, 2, 4, 4
outC = 2
kH, kW = 3, 3

def example_dconv():
    from dcn_v2 import DCN
    torch.random.manual_seed(1234)
    input = torch.randn(1, 64, 128, 128).cuda()
    offset = torch.randn(1, 2 * 3 * 3, 128, 128).cuda()
    mask = torch.randn(1, 1 * 3 * 3, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    dcn = DCNv2(64, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1).cuda()
    
    output = dcn(input, offset, mask)
    #targert = output.new(*output.size())
    #targert.data.uniform_(-0.01, 0.01)
    #error = (targert - output).mean()
    #error.backward()
    print(output.shape)
    print(output)

    print('weight: ', dcn.weight.shape)
    print('weight: ', dcn.weight)
    print('bias: ', dcn.bias.shape)

    import struct

    f = open("dcnv2.wts", 'w')
    
    dict = {'input': input, 
        'dcnv2.weight': dcn.weight,
        'dcnv2.bias': dcn.bias,
        'offset': offset,
        'mask': mask}

    f.write("{}\n".format(len(dict.keys())))
    for k,v in dict.items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().detach().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

if __name__ == '__main__':

    example_dconv()
