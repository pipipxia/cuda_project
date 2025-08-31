import torch
import numpy as np
import time
import functools
from typing import Dict, Type, Any, Optional, Union, Callable, Optional


def timer(description: Optional[str] = None) -> Callable:
    """
    计时装饰器函数（支持自定义描述）。

    用法：
    @timer("矩阵乘法")
    def foo(...):
        ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            tag = description if description is not None else func.__name__
            print(f"[Timer] {tag} took {elapsed*1000:.6f} ms")
            return result
        return wrapper
    return decorator

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

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
        print(f"test_data:{test_data}")
        print(f"golden_data:{golden_data}")
        assert 0, "compare_data failed"
    else:
        print(f"data compare success!")
    
    