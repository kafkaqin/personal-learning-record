import torch
import numpy as np
import torch.nn.functional as F

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    x = torch.tensor([[-1.0, 0.0, 1.0],[-1.0, 0.0, 1.0]])
    softmax_result= F.softmax(x,dim=1)
    print(softmax_result)


if __name__ == '__main__':
    main()
    y = np.array([[-1.0, 0.0, 1.0],[-1.0, 0.0, 1.0]])
    x = sigmoid(y)
    print(x)