import torch
from python import *
import cuda_op

def test_elemwise():
<<<<<<< HEAD
    shape = (256, 256)
    tensor0 = torch.rand(shape)
    tensor1 = torch.rand(shape)
    test_data = torch.rand(shape)
=======
    shape = (16, 16)
    tensor0 = torch.rand(shape, dtype=torch.float64)
    tensor1 = torch.rand(shape, dtype=torch.float64)
    test_data = torch.rand(shape, dtype=torch.float64)
>>>>>>> 261b8c4 (cuda project init)
    
    # python
    golden_data = elemwise_add(tensor0, tensor1)
    
    # cuda
<<<<<<< HEAD
    cuda_op.elemwise_add(test_data, tensor0, tensor1)
=======
    elemwise = cuda_op.ElemWise(test_data, tensor0, tensor1)
    elemwise.add()
>>>>>>> 261b8c4 (cuda project init)
    
    compare_data(test_data, golden_data, max_diff = 0)
    
if __name__ == "__main__":
    test_elemwise()