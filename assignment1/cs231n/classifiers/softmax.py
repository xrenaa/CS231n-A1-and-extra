import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
      score=X[i].dot(W)
      score-=np.max(score)
      sum_score=np.sum(np.exp(score))
      p = lambda k : np.exp(score[k]) / sum_score
      loss += -np.log(p(y[i]))
      for j in range(W.shape[1]):
        p_j=p(j)
        if y[i]==j:
            dW[:,j]+=(-1 + p_j)*X[i]
        else:
            dW[:,j]+=p_j*X[i]
    
  
  loss /= (X.shape[0])
  loss += 0.5 * reg * np.sum(W*W)

  dW/= (X.shape[0])
  dW+= reg*W
      
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N=X.shape[0]
  D=X.shape[1]
  C=W.shape[1]
  
  score=X.dot(W)
  score-=np.max(score,axis=1,keepdims=True)
  
  p= np.exp(score)/np.sum(np.exp(score),axis=1,keepdims=True)
  #print('p shape: ', p.shape)
  #print('W shape: ', W.shape)
  #print('X shape: ', X.shape)
  #print('score shape: ', score.shape)
  
  y_map=np.zeros((N,C))
  y_map[np.arange(N),y]=1
  
  loss = -1 * np.sum(np.multiply(np.log(p),y_map)) / N
  loss += 0.5 * reg * np.sum( W * W)
  
  dW = X.T.dot(p-y_map)
  dW /= N
  dW += reg*W

  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

