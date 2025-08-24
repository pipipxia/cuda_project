import torch
from python import *
import cuda_op

def test_elemwise():
    shape = (256, 256)
    tensor0 = torch.rand(shape)
    tensor1 = torch.rand(shape)
    test_data = torch.rand(shape)
    
    # python
    golden_data = elemwise_add(tensor0, tensor1)
    
    # cuda
    cuda_op.elemwise_add(test_data, tensor0, tensor1)
    
    compare_data(test_data, golden_data, max_diff = 0)
    
if __name__ == "__main__":
    test_elemwise()