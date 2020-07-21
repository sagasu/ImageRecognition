from __future__ import print_function
import torch

#whatever values where in that memory will be in that matrix
x = torch.empty(5, 3)
print(x)


#random
x = torch.rand(5, 3)
print(x)

#zeros, dtype long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)


#tensor from data
x = torch.tensor([5.5, 3])
print(x)

#tensor from tensor, can change type
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

#get size
print(x.size())