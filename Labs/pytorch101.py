import torch

# Type hints.
from typing import List, Tuple
from torch import Tensor


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from pytorch101.py!')


def create_sample_tensor() -> Tensor:
    """
    Return a torch Tensor of shape (3, 2) which is filled with zeros, except
    for element (0, 1) which is set to 10 and element (1, 0) which is set to
    100.

    Returns:
        Tensor of shape (3, 2) as described above.
    """
    x = None
    ##########################################################################
    #                     TODO: Implement this function                      #
    ##########################################################################
    # Replace "pass" statement with your code
    pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return x


def mutate_tensor(
    x: Tensor, indices: List[Tuple[int, int]], values: List[float]
) -> Tensor:
    """
    Mutate the tensor x according to indices and values. Specifically, indices
    is a list [(i0, j0), (i1, j1), ... ] of integer indices, and values is a
    list [v0, v1, ...] of values. This function should mutate x by setting:

    x[i0, j0] = v0
    x[i1, j1] = v1

    and so on.

    If the same index pair appears multiple times in indices, you should set x
    to the last one.

    Args:
        x: A Tensor of shape (H, W)
        indices: A list of N tuples [(i0, j0), (i1, j1), ..., ]
        values: A list of N values [v0, v1, ...]

    Returns:
        The input tensor x
    """
    ##########################################################################
    #                     TODO: Implement this function                      #
    ##########################################################################
    # Replace "pass" statement with your code
    pass
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def count_tensor_elements(x: Tensor) -> int:
    """
    Count the number of scalar elements in a tensor x.

    For example, a tensor of shape (10,) has 10 elements; a tensor of shape
    (3, 4) has 12 elements; a tensor of shape (2, 3, 4) has 24 elements, etc.

    You may not use the functions torch.numel or x.numel. The input tensor
    should not be modified.

    Args:
        x: A tensor of any shape

    Returns:
        num_elements: An integer giving the number of scalar elements in x
    """
    num_elements = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    #   You CANNOT use the built-in functions torch.numel(x) or x.numel().   #
    ##########################################################################
    # Replace "pass" statement with your code
    pass
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return num_elements


def create_tensor_of_pi(M: int, N: int) -> Tensor:
    """
    Returns a Tensor of shape (M, N) filled entirely with the value 3.14

    Args:
        M, N: Positive integers giving the shape of Tensor to create

    Returns:
        x: A tensor of shape (M, N) filled with the value 3.14
    """
    x = None
    ##########################################################################
    #         TODO: Implement this function. It should take one line.        #
    ##########################################################################
    # Replace "pass" statement with your code
    pass
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def multiples_of_ten(start: int, stop: int) -> Tensor:
    """
    Returns a Tensor of dtype torch.float64 that contains all of the multiples
    of ten (in order) between start and stop, inclusive. If there are no
    multiples of ten in this range then return an empty tensor of shape (0,).

    Args:
        start: Beginning ot range to create.
        stop: End of range to create (stop >= start).

    Returns:
        x: float64 Tensor giving multiples of ten between start and stop
    """
    assert start <= stop
    x = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    pass
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x