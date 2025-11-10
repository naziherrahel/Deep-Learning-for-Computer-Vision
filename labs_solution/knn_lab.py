
"""
Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import torch
from typing import Dict, List


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from knn.py!")


def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses a naive set of nested loops over the training and
    test data.

    The input data may have any number of dimensions -- for example this
    function should be able to compute nearest neighbor between vectors, in
    which case the inputs will have shape (num_{train, test}, D); it should
    also be able to compute nearest neighbors between images, where the inputs
    will have shape (num_{train, test}, C, H, W). More generally, the inputs
    will have shape (num_{train, test}, D1, D2, ..., Dn); you should flatten
    each element of shape (D1, D2, ..., Dn) into a vector of shape
    (D1 * D2 * ... * Dn) before computing distances.

    The input tensors should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants (`x.norm`, `x.dist`,
    `x.cdist`, etc.). You may not use any functions from `torch.nn` or
    `torch.nn.functional` modules.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It should have the same dtype as x_train.
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    print("TWO-LOOP - Input checksum:")
    print(f"x_train: {x_train.sum().item()}, x_test: {x_test.sum().item()}")
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    # dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function using a pair of nested loops over the    #
    # training data and the test data.                                       #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    ##########################################################################
    # Replace "pass" statement with your code
    # Flatten the training and test tensors
    # Use reshape for safe flattening
    # Flatten with explicit contiguous handling
    # Flatten with explicit contiguous handling
    
    
    # Flatten using flatten() for guaranteed consistency
    x_train_flat = x_train.flatten(start_dim=1)
    x_test_flat = x_test.flatten(start_dim=1)
    
    dists = x_train.new_zeros(num_train, num_test)
    
    for i in range(num_train):
        for j in range(num_test):
            # Direct element-wise operations
            diff = x_train_flat[i] - x_test_flat[j]
            sq_diff = diff * diff
            dists[i, j] = sq_diff.sum()
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses only a single loop over the training data.

    Similar to `compute_distances_two_loops`, this should be able to handle
    inputs with any number of dimensions. The inputs should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants (`x.norm`, `x.dist`,
    `x.cdist`, etc.). You may not use any functions from `torch.nn` or
    `torch.nn.functional` modules.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It should have the same dtype as x_train.
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    print("ONE-LOOP - Input checksum:")
    print(f"x_train: {x_train.sum().item()}, x_test: {x_test.sum().item()}")
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    # dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function using only a single loop over x_train.   #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    ##########################################################################
    # Replace "pass" statement with your code
    # Flatten the training and test tensors
    # Use reshape for safe flattening
    # Flatten with explicit contiguous handling
    
    
    # Flatten using same method as two-loop version
    x_train_flat = x_train.flatten(start_dim=1)
    x_test_flat = x_test.flatten(start_dim=1)
    
    dists = x_train.new_zeros(num_train, num_test)
    
    for i in range(num_train):
        # Compute differences with explicit broadcasting
        train_vec = x_train_flat[i].unsqueeze(0)  # Shape: (1, features)
        diff = train_vec - x_test_flat  # Broadcasts to (num_test, features)
        
        # Same operations as two-loop version
        sq_diff = diff * diff
        dists[i] = sq_diff.sum(dim=1)  # Sum along feature dimension

    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation should not use any Python loops. For memory-efficiency,
    it also should not create any large intermediate tensors; in particular you
    should not create any intermediate tensors with O(num_train * num_test)
    elements.

    Similar to `compute_distances_two_loops`, this should be able to handle
    inputs with any number of dimensions. The inputs should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants (`x.norm`, `x.dist`,
    `x.cdist`, etc.). You may not use any functions from `torch.nn` or
    `torch.nn.functional` modules.

    Args:
        x_train: Tensor of shape (num_train, C, H, W)
        x_test: Tensor of shape (num_test, C, H, W)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j] is
            the squared Euclidean distance between the i-th training point and
            the j-th test point.
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function without using any explicit loops and     #
    # without creating any intermediate tensors with O(num_train * num_test) #
    # elements.                                                              #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    #                                                                        #
    # HINT: Try to formulate the Euclidean distance using two broadcast sums #
    #       and a matrix multiply.                                           #
    ##########################################################################
    # Replace "pass" statement with your code
    # Flatten inputs
    x_train_flat = x_train.flatten(start_dim=1)  # Shape: (num_train, features)
    x_test_flat = x_test.flatten(start_dim=1)    # Shape: (num_test, features)
    
    # Compute squared Euclidean distance: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
    # ||x||^2: Sum of squares for each training point
    train_norms = (x_train_flat * x_train_flat).sum(dim=1)  # Shape: (num_train,)
    
    # ||y||^2: Sum of squares for each test point
    test_norms = (x_test_flat * x_test_flat).sum(dim=1)     # Shape: (num_test,)
    
    # -2 * x^T * y: Matrix multiply for dot products
    dot_products = torch.matmul(x_train_flat, x_test_flat.T)  # Shape: (num_train, num_test)
    
    # Broadcast: train_norms[:, None] becomes (num_train, 1), test_norms becomes (1, num_test)
    dists = train_norms[:, None] + test_norms - 2 * dot_products
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


