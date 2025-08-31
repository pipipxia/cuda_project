import torch
from python import *
import cuda_op

@timer(f"[python] elemwise_add")
def elemwise_add(tensor0: torch.Tensor, tensor1:torch.Tensor):
    return tensor0 + tensor1

@timer(f"[cuda] elemwise_add")
def elemwise_add_cuda(tensor0: torch.Tensor, tensor1:torch.Tensor):
    test_data = torch.empty_like(tensor0)
    if tensor0.dtype == torch.float64:
        elemwise = cuda_op.ElemWiseFloat64(test_data.numpy(), tensor0.numpy(), tensor1.numpy())
        elemwise.add()
    elif tensor0.dtype == torch.float32:
        elemwise = cuda_op.ElemWiseFloat32(test_data.numpy(), tensor0.numpy(), tensor1.numpy())
        elemwise.add()
    return test_data