from __future__ import print_function
import torch


x = torch.rand(5, 3)
y = torch.rand(5, 3)

print(x + y)

# same as the one above
print(torch.add(x, y))


result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)


# adds x to y
y.add_(x)
print(y)

print(x[:, 1])

# view method is for resizing/reshaping tensor
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())


x = torch.randn(1)
print(x)
print(x.item())