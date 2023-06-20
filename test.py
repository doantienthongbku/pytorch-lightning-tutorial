import torch
import torch.nn as nn


def test():
    weight = torch.randn(4, 64, 32, 3, 3)   # (E, Cout, Cin, k, k)
    weight = weight.view(4, 64, -1) # (E, Cout, Cin * k * k)
    weight = nn.Parameter(weight, requires_grad=True)
    coeff = torch.randn(2, 64, 4) # (bs, Cout, E)
    
    aggregate_weight = torch.zeros(2, 64, 32, 3, 3)    # (bs, Cout, Cin, k, k)
    
    for i in range(64):
        sub_weight = weight[:, i, :] # (E, Cin * k * k)
        sub_coeff = coeff[:, i, :] # (bs, E)
        print(type(sub_weight), type(sub_coeff))
        sub_aggregate_weight = torch.mm(sub_coeff, sub_weight) # (bs, Cin * k * k)
        aggregate_weight[:, i, :] = sub_aggregate_weight.view(2, 32, 3, 3)
    
    aggregate_weight = aggregate_weight.view(2 * 64, 32, 3, 3)  # (bs * Cout, Cin, k, k)
    # print(aggregate_weight.shape)
    
    input = torch.randn(2, 32, 64, 64)
    input = input.view(1, 64, 64, 64)
    out = F.conv2d(input, weight=aggregate_weight, bias=None, stride=1, padding=1, dilation=1, groups=2)
    out = out.view(2, 64, 64, 64)
    # print("input shape: ", input.shape)
    # print("out shape: ", out.shape)