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

