
import numpy as np


def num_elements_gb(dtype, available_memory_gb):
    """
    Calculate the number of elements that fit into the available memory.

    Parameters:
    dtype (data-type): Data type of the variable.
    available_memory_gb (float): Available memory in gigabytes.

    Returns:
    int: Number of elements that fit into the available memory.
    """
    # Convert available memory from GB to bytes
    available_memory_bytes = available_memory_gb * 1024**3

    # Calculate the size of a single element in bytes
    element_size_bytes = np.dtype(dtype).itemsize

    # Calculate the number of elements that fit into the available memory
    num_elements = available_memory_bytes // element_size_bytes
    print(f"Number of elements that fit into the available memory: {num_elements}")
    return int(num_elements)

def num_elements_mb(dtype, available_memory_mb):
    """
    Calculate the number of elements that fit into the available memory.

    Parameters:
    dtype (data-type): Data type of the variable.
    available_memory_gb (float): Available memory in gigabytes.

    Returns:
    int: Number of elements that fit into the available memory.
    """
    # Convert available memory from GB to bytes
    available_memory_bytes = available_memory_mb * 1024 ** 2

    # Calculate the size of a single element in bytes
    element_size_bytes = np.dtype(dtype).itemsize

    # Calculate the number of elements that fit into the available memory
    num_elements = available_memory_bytes // element_size_bytes
    print(f"Number of elements that fit into the available memory: {num_elements}")
    return int(num_elements)


def get_diag_mask(shape):
    mask = np.zeros((shape), dtype=bool)
    for i in range(shape[0]):
        for j in range(shape[2]):
            mask[i, i, j, j] = True
    return mask


def einsum(equation, *operands):
    return np.einsum(equation, *operands)
