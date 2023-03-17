from itertools import product
from typing import Any, Dict, List, Tuple


def create_product_parametrize(params: Dict[str, List[Any]]) -> Tuple[str, List[List[Any]]]:
    names = []
    args = []
    for key, value in params.items():
        names.append(key)
        args.append(value)
    return ",".join(names), list(product(*args))


def compute_conv_size(
    size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
