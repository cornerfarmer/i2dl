"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    y_dev = y
    from math import exp
    y_log = np.zeros([y_dev.shape[0], W.shape[1]])
    y = np.zeros([y_dev.shape[0], W.shape[1]])
    for t in range(y_dev.shape[0]):
        y_sum = 0
        for i in range(y.shape[1]):
            for j in range(W.shape[0]):
                y_log[t][i] += W[j][i] * X[t][j]
            y[t][i] = exp(y_log[t][i])
            y_sum += y[t][i] 

        for i in range(y.shape[1]):
            y[t][i] /= y_sum  
            
            
            
    from math import log
    loss = 0.0
    for t in range(y.shape[0]):
        for i in range(y.shape[1]):
            if i == y_dev[t]:
                loss -= log(y[t][i])
    loss /= y.shape[0]
    
    w_sum = 0
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w_sum += W[i][j] ** 2
    loss += reg * w_sum / (W.shape[0] * W.shape[1])
    
    dLoss = np.zeros_like(y)
    for t in range(y.shape[0]):
        for i in range(y.shape[1]):
            if i == y_dev[t]:
                dLoss[t][i] = -1 / y[t][i]
            dLoss[t][i] /= y.shape[0]
                
                
                
                
    dY = np.zeros_like(y)
    for t in range(y.shape[0]):
        y_sum = 0
        for i in range(y.shape[1]):
            y_sum += exp(y_log[t][i])

        for i in range(y.shape[1]):
            for j in range(y.shape[1]):
                dY[t][i] += exp(y_log[t][j]) * dLoss[t][j] * -1 / (y_sum ** 2)
        
            dY[t][i] += dLoss[t][i] * 1 / y_sum
            dY[t][i] *= exp(y_log[t][i])

            
            
            
    dW = np.zeros_like(W)
    for t in range(y.shape[0]):
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                dW[i][j] += X[t][i] * dY[t][j]
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            dW[i][j] += reg * 2 / (W.shape[0] * W.shape[1]) * W[i][j]

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    y_log = np.matmul(X, W)
    y_pred = np.argmax(y_log, -1)
    y_exp = np.exp(y_log - np.expand_dims(np.max(y_log, -1), -1))
    y_sum = np.expand_dims(np.sum(y_exp, -1), axis=-1)
    y_act = y_exp / y_sum
    
    y_one_hot = np.eye(W.shape[1])[y]
    loss = np.mean(np.sum(-y_one_hot * np.log(y_act), -1))
    loss += reg * np.sum(W ** 2)
    
    dLoss = y_one_hot * -1 / y_act
    dLoss /= y.shape[0]
    
    dY_base = np.sum(y_exp * dLoss * -1 / (y_sum ** 2), -1)
    
    dY = (dLoss / y_sum + np.expand_dims(dY_base, axis=-1)) * y_exp
       
    
    dW = np.matmul(np.transpose(X), dY)
    dW += reg * 2 / (W.shape[0] * W.shape[1]) * W

    acc = np.mean(y == y_pred)

    return loss, acc, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)
