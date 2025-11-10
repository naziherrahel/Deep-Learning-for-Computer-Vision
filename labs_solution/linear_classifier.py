"""
Implements linear classifeirs in PyTorch.
"""
import torch
import random
import statistics
from abc import abstractmethod
from typing import Dict, List, Callable, Optional


def hello_linear_classifier():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from linear_classifier.py!")


# Template class modules that we will use later: Do not edit/modify this class
class LinearClassifier:
    """An abstarct class for the linear classifiers"""

    # Note: We will re-use `LinearClassifier' in both SVM and Softmax
    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W = None

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        train_args = (
            self.loss,
            self.W,
            X_train,
            y_train,
            learning_rate,
            reg,
            num_iters,
            batch_size,
            verbose,
        )
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X: torch.Tensor):
        return predict_linear_classifier(self.W, X)

    @abstractmethod
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
        - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an tensor of the same shape as W
        """
        raise NotImplementedError

    def _loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.W, X_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({"W": self.W}, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        W_dict = torch.load(path, map_location="cpu")
        self.W = W_dict["W"]
        if self.W is None:
            raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)


# **************************************************#
################## Section 1: SVM ##################
# **************************************************#


def svm_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples. When you implment the regularization over W, please DO NOT
    multiply the regularization term by 1/2 (no coefficient).

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    # Initialize the gradient tensor to zeros (same shape as W)
    dW = torch.zeros_like(W)

    # Extract dimensions
    num_classes = W.shape[1]  # C — number of output classes
    num_train = X.shape[0]    # N — number of training examples
    loss = 0.0                # Initialize total loss

    # Loop over each training example
    for i in range(num_train):
        # Compute the class scores for this sample
        # (Each score corresponds to a dot product between input X[i] and column W[:, j])
        scores = W.t().mv(X[i])  # (C,) = (C, D) @ (D,)
        
        # Get the score of the correct class
        correct_class_score = scores[y[i]]
        
        # Loop over all classes to compute margins
        for j in range(num_classes):
            # Skip the correct class — it does not contribute to margin
            if j == y[i]:
                continue
            
            # Compute margin for incorrect class j
            # margin = s_j - s_yi + delta, with delta = 1
            margin = scores[j] - correct_class_score + 1  
            
            # Apply hinge loss rule: only positive margins contribute
            if margin > 0:
                # Accumulate total loss
                loss += margin
                #######################################################################
                # TODO:                                                               #
                # Compute the gradient of the SVM term of the loss function and store #
                # it on dW. (part 1) Rather than first computing the loss and then    #
                # computing the derivative, it is simple to compute the derivative    #
                # at the same time that the loss is being computed.                   #
                #######################################################################
                # For incorrect class (j ≠ y[i]): ∂L/∂W_j = X[i]
                dW[:, j] += X[i]
                # For correct class (y[i]): ∂L/∂W_y[i] = -X[i]
                dW[:, y[i]] -= X[i]
                #######################################################################
                #                       END OF YOUR CODE                              #
                #######################################################################

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # Average the gradient as well
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * torch.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function w.r.t. the regularization term  #
    # and add it to dW. (part 2)                                                #
    #############################################################################
    # Replace "pass" statement with your code
    dW += 2 * reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW

def svm_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, vectorized implementation. When you implement
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    #############################################################################
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape[0]  # number of training samples
    
    # Compute scores for all samples and classes:
    # Each row i gives the scores for all classes of sample i
    scores = X @ W  # (N, C) = (N, D) @ (D, C)
    
    # Extract the correct class scores using label y for each training example
    # This gives a vector of size (N,)
    correct_class_scores = scores[range(num_train), y]
    
    # Compute the margins for all classes:
    # margin(i, j) = score(i, j) - score(i, y_i) + 1
    # Broadcasting correct_class_scores as a column vector
    margins = scores - correct_class_scores.view(-1, 1) + torch.tensor(1.0, dtype=X.dtype)
    
    # The margin of the correct class is always zero by definition
    margins[range(num_train), y] = 0
    
    # Apply hinge loss: only keep positive margins, others become zero
    margins = torch.clamp(margins, min=0)
    
    # Compute total loss: sum of all positive margins divided by number of samples
    loss = margins.sum() / num_train
    
    # Add regularization term to prevent overfitting
    # (no 1/2 factor as per instruction)
    loss += reg * torch.sum(W * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #############################################################################
    # Create a binary mask indicating where margins are positive
    # This tells us which classes contributed to the loss
    binary = (margins > 0).to(X.dtype)
    
    # Compute the gradient contribution for incorrect classes:
    # For every (i, j) with positive margin, we add X[i] to dW[:, j]
    dW = X.t() @ binary / num_train  # (D, C) = (D, N) @ (N, C)
    
    # For the correct classes, the gradient is the negative sum of the
    # contributions from all positive incorrect classes for each sample
    correct_class_grad = -binary.sum(dim=1)  # (N,), each entry = -# of positive margins for that sample
    
    # Build a one-hot mask where each sample i has a 1 at column y[i]
    correct_class_mask = torch.zeros_like(scores).scatter_(1, y.unsqueeze(1), 1)
    
    # Multiply the mask by the per-sample gradient magnitude
    # This applies the negative count only to the correct class column
    correct_class_grad = correct_class_grad.view(-1, 1) * correct_class_mask
    
    # Add this correct-class gradient to dW
    # This subtracts the correct class contributions for each sample
    dW += X.t() @ correct_class_grad / num_train
    
    # Finally, add the gradient of the regularization term
    # derivative of reg * ||W||^2 = 2 * reg * W
    dW += 2 * reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW

def sample_batch(
    X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int
):
    """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.
    """
    X_batch = None
    y_batch = None
    #########################################################################
    # TODO: Store the data in X_batch and their corresponding labels in     #
    # y_batch; after sampling, X_batch should have shape (batch_size, dim)  #
    # and y_batch should have shape (batch_size,)                           #
    #                                                                       #
    # Hint: Use torch.randint to generate indices.                          #
    #########################################################################
    # Generate random indices for sampling
    indices = torch.randint(0, num_train, (batch_size,), device=X.device)
    # Sample from X and y using the indices
    X_batch = X[indices]  # Shape: (batch_size, D)
    y_batch = y[indices]  # Shape: (batch_size,)
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return X_batch, y_batch



def train_linear_classifier(
    loss_func: Callable,
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 1e-3,
    reg: float = 1e-5,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
      """
      Train this linear classifier using stochastic gradient descent.

      Inputs:
      - loss_func: loss function to use when training. It should take W, X, y
        and reg as input, and output a tuple of (loss, dW)
      - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
        classifier. If W is None then it will be initialized here.
      - X: A PyTorch tensor of shape (N, D) containing training data; there are N
        training samples each of dimension D.
      - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
        means that X[i] has label 0 <= c < C for C classes.
      - learning_rate: (float) learning rate for optimization.
      - reg: (float) regularization strength.
      - num_iters: (integer) number of steps to take when optimizing
      - batch_size: (integer) number of training examples to use at each step.
      - verbose: (boolean) If true, print progress during optimization.

      Returns: A tuple of:
      - W: The final value of the weight matrix and the end of optimization
      - loss_history: A list of Python scalars giving the values of the loss at each
        training iteration.
      """
      # assume y takes values 0...K-1 where K is number of classes
      num_train, dim = X.shape
      if W is None:
          # lazily initialize W
          num_classes = torch.max(y) + 1
          W = 0.000001 * torch.randn(
              dim, num_classes, device=X.device, dtype=X.dtype
          )
      else:
          num_classes = W.shape[1]
      # Run stochastic gradient descent to optimize W
      loss_history = []
      for it in range(num_iters):
          X_batch, y_batch = sample_batch(X, y, num_train, batch_size)
          loss, grad = loss_func(W, X_batch, y_batch, reg)
          loss_history.append(loss.item())

          # perform parameter update
          #########################################################################
          # TODO:                                                                 #
          # Update the weights using the gradient and the learning rate.          #
          #########################################################################
          W -= learning_rate * grad
          #########################################################################
          #                       END OF YOUR CODE                                #
          #########################################################################

          if verbose and it % 100 == 0:
              print("iteration %d / %d: loss %f" % (it, num_iters, loss))

      return W, loss_history


def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        """
        y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # Compute scores: X @ W gives (N, C) matrix of scores for each class
        scores = X @ W
        # Predict the class with the highest score for each sample
        y_pred = torch.argmax(scores, dim=1)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

def svm_get_search_params():
      """
      Return candidate hyperparameters for the SVM model.
      """
      learning_rates = []
      regularization_strengths = []

      ###########################################################################
      # TODO:   add your own hyper parameter lists.                             #
      ###########################################################################
      learning_rates = [1e-3, 1e-2, 1e-1]
      regularization_strengths = [1e-3, 1e-2, 1e-1, 1e0]
      ###########################################################################
      #                           END OF YOUR CODE                              #
      ###########################################################################

      return learning_rates, regularization_strengths

def test_one_param_set(
    cls: LinearClassifier,
    data_dict: Dict[str, torch.Tensor],
    lr: float,
    reg: float,
    num_iters: int = 2000,
):
    """
    Train a single LinearClassifier instance and return the learned instance
    with train/val accuracy.
    """
    train_acc = 0.0
    val_acc = 0.0
    ###########################################################################
    # TODO:                                                                   #
    # Write code that, train a linear SVM on the training set, compute its    #
    # accuracy on the training and validation sets                            #
    ###########################################################################
    # Train the classifier
    cls.train(
        data_dict['X_train'],
        data_dict['y_train'],
        learning_rate=lr,
        reg=reg,
        num_iters=num_iters,
        batch_size=200,
        verbose=False
    )
    
    # Predict on training set
    y_train_pred = cls.predict(data_dict['X_train'])
    train_acc = (y_train_pred == data_dict['y_train']).float().mean().item()
    
    # Predict on validation set
    y_val_pred = cls.predict(data_dict['X_val'])
    val_acc = (y_val_pred == data_dict['y_val']).float().mean().item()
    ############################################################################
    #                            END OF YOUR CODE                              #
    ############################################################################

    return cls, train_acc, val_acc

# **************************************************#
################ Section 2: Softmax ################
# **************************************************#


def softmax_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    
    pass
  

    return None


def softmax_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    pass

    return None


def softmax_get_search_params():
   

    return None
