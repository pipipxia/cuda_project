import torch
from python import *
import cuda_op

def test_elemwise():
    device = 'cpu'# get_device()
    shape = (32, 32)
    dtype = torch.float32
    tensor0 = torch.rand(shape, dtype=dtype, device=device)
    tensor1 = torch.rand(shape, dtype=dtype, device=device)
    
    # python
    golden_data = elemwise_add(tensor0, tensor1)
    
    # cuda
    test_data = elemwise_add_cuda(tensor0, tensor1)
    
    compare_data(test_data, golden_data, max_diff = 0)
    
if __name__ == "__main__":
    test_elemwise()