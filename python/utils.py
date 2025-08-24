import torch
import numpy as np

def Numpy2TensorData(np_array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np_array)
    
def Tensor2NumpyData(tensor: torch.Tensor) -> np.ndarray:
    if tensor.device.type == 'cpu':
        return tensor.numpy()
    else:
        return tensor.cpu().numpy()

def compare_data(test_data, golden_data, max_diff = 0):
    if isinstance(test_data, np.ndarray):
        test_data = Numpy2TensorData(test_data)
    if isinstance(golden_data, np.ndarray):
        golden_data = Numpy2TensorData(golden_data)
    test_data = test_data.reshape(-1)
    golden_data = golden_data.reshape(-1)
    diff = torch.abs(golden_data - test_data)
    count = (diff > max_diff).sum().item()
    if count > 0:
        print(f"data compare failed!")
        print(f"missmatch num:{count}:{count/golden_data.numel():.3%}")
        print(f"max diff:{torch.max(diff)}")
        assert 0
    else:
        print(f"data compare success!")
    