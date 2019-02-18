from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    C=W2.shape[1]
    layer_1=np.dot(X,W1)+b1
    #reLu
    layer_1_relu=layer_1
    layer_1_relu[layer_1_relu<0]=0
    #output layer
    out_layer=np.dot(layer_1_relu,W2)+b2
    
    scores=out_layer
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    scores-=np.max(scores,axis=1,keepdims=True)
    
    p= np.exp(scores)/np.sum(np.exp(scores),axis=1,keepdims=True)
    
    
    y_map=np.zeros((N,C))
    y_map[np.arange(N),y]=1
  
    loss = -1 * np.sum(np.multiply(np.log(p),y_map)) / N
    loss += reg *(np.sum(W1 * W1)+np.sum(W2*W2)+np.sum(b1*b1)+np.sum(b2*b2))
                  
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    dscores=p
    dscores[range(N),y]-=1
    dscores/=N
    dW2=np.dot(layer_1_relu.T,dscores)
    db2=np.sum(dscores,axis=0,keepdims=False)
    
    dh1=np.dot(dscores,W2.T)
    
    #RELU layer
    dh1[layer_1_relu<=0]=0
    dW1=np.dot(X.T,dh1)
    db1=np.sum(dh1,axis=0,keepdims=False)
    
    #add regularization 
    dW2+=2*reg*W2
    dW1+=2*reg*W1
    
    #result
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2              
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      batch_index=np.random.choice(num_train,batch_size)
      X_batch=X[batch_index,:]
      y_batch=y[batch_index]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1']+= -grads['W1']*learning_rate
      self.params['W2']+= -grads['W2']*learning_rate
      self.params['b1']+= -grads['b1']*learning_rate
      self.params['b2']+= -grads['b2']*learning_rate
      
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }
#   def train(self, X, y, X_val, y_val,
#             learning_rate=0.01, beta1=0.9,beta2=0.999,epsilon=1e-8,
#             reg=5e-6, num_iters=100,
#             batch_size=200, verbose=False):
#     """
#     Train this neural network using stochastic gradient descent.

#     Inputs:
#     - X: A numpy array of shape (N, D) giving training data.
#     - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
#       X[i] has label c, where 0 <= c < C.
#     - X_val: A numpy array of shape (N_val, D) giving validation data.
#     - y_val: A numpy array of shape (N_val,) giving validation labels.
#     - learning_rate: Scalar giving learning rate for optimization.
#     - learning_rate_decay: Scalar giving factor used to decay the learning rate
#       after each epoch.
#     - reg: Scalar giving regularization strength.
#     - num_iters: Number of steps to take when optimizing.
#     - batch_size: Number of training examples to use per step.
#     - verbose: boolean; if true print progress during optimization.
#     """
#     num_train = X.shape[0]
#     iterations_per_epoch = max(num_train / batch_size, 1)
    
#     loss_history = []
#     train_acc_history = []
#     val_acc_history = []


#     #initialize adam
#     L = 2 # number of layers in the neural networks
#     v = {}
#     s = {}
#     t = 0
#     v_corrected = {}       # Initializing first moment estimate, python dictionary
#     s_corrected = {}       # Initializing second moment estimate, python dictionary
#     # Initialize v, s. Input: "parameters". Outputs: "v, s".
#     for l in range(L):
#     ### START CODE HERE ### (approx. 4 lines)
#         v["dW" + str(l + 1)] = np.zeros_like(self.params["W" + str(l + 1)])
#         v["db" + str(l + 1)] = np.zeros_like(self.params["b" + str(l + 1)])

#         s["dW" + str(l+1)] = np.zeros_like(self.params["W" + str(l + 1)])
#         s["db" + str(l+1)] = np.zeros_like(self.params["b" + str(l + 1)])
    
    
#     for it in xrange(num_iters):
#       t = t+1
#       X_batch = None
#       y_batch = None

     
#       batch_index=np.random.choice(num_train,batch_size)
#       X_batch=X[batch_index,:]
#       y_batch=y[batch_index]
      

#       # Compute loss and gradients using the current minibatch
#       loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
#       loss_history.append(loss)
    
# #       self.params['W1']+= -grads['W1']*learning_rate
# #       self.params['W2']+= -grads['W2']*learning_rate
# #       self.params['b1']+= -grads['b1']*learning_rate
# #       self.params['b2']+= -grads['b2']*learning_rate
      
#       for l in range(L):
#         # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
#         ### START CODE HERE ### (approx. 2 lines)
#         v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['W' + str(l + 1)]
#         v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['b' + str(l + 1)]

#         # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
#         v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
#         v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

#         # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
#         ### START CODE HERE ### (approx. 2 lines)
#         s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['W' + str(l + 1)], 2)
#         s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['b' + str(l + 1)], 2)
#         ### END CODE HERE ###

#         # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        
#         s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
#         s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
       
#         # Update parameters.Output: "parameters".
#         self.params["W" + str(l + 1)] = self.params["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)
#         self.params["b" + str(l + 1)] = self.params["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon)
        
        
        
#       if verbose and it % 100 == 0:
#         print('iteration %d / %d: loss %f' % (it, num_iters, loss))

#       # Every epoch, check train and val accuracy and decay learning rate.
#       if it % iterations_per_epoch == 0:
#         # Check accuracy
#         train_acc = (self.predict(X_batch) == y_batch).mean()
#         val_acc = (self.predict(X_val) == y_val).mean()
#         train_acc_history.append(train_acc)
#         val_acc_history.append(val_acc)


#     return {
#       'loss_history': loss_history,
#       'train_acc_history': train_acc_history,
#       'val_acc_history': val_acc_history,
#     }
# def train(self, X, y, X_val, y_val, learning_rate=1e-3, 
#                learning_rate_decay=0.95, reg=1e-5, mu=0.9, num_epochs=10, 
#                mu_increase=1.0, batch_size=200, verbose=False):   
#         """    
#         还是使用SGD 
#         Inputs:    
#         - X:  (N, D)    
#         - y:(N,) 
#         - X_val:  (N_val, D)     
#         - y_val:  (N_val,)     
#         - learning_rate:     
#         - learning_rate_decay:   学习率衰减因子
#         - reg:    
#         - num_iters:    
#         - batch_size:     
#         - verbose: boolean; if true print progress during optimization.  
#         """
#         num_train = X.shape[0]
#         iterations_per_epoch = max(num_train / batch_size, 1)
        
#         v_W2, v_b2 = 0.0, 0.0
#         v_W1, v_b1 = 0.0, 0.0
#         loss_history = []
#         train_acc_history = []
#         val_acc_history = []

#         for it in xrange(1, num_epochs * iterations_per_epoch + 1):   
#             X_batch = None   
#             y_batch = None    
               
#             sample_index = np.random.choice(num_train, batch_size, replace=True)   
#             X_batch = X[sample_index, :]        # (batch_size,D)    
#             y_batch = y[sample_index]           # (1,batch_size)   

           
#             loss, grads = self.loss(X_batch, y=y_batch, reg=reg) 
#             loss_history.append(loss)    

#             # SGD结合向量    
#             v_W2 = mu * v_W2 - learning_rate * grads['W2']    
#             self.params['W2'] += v_W2   
#             v_b2 = mu * v_b2 - learning_rate * grads['b2']    
#             self.params['b2'] += v_b2   
#             v_W1 = mu * v_W1 - learning_rate * grads['W1']    
#             self.params['W1'] += v_W1   
#             v_b1 = mu * v_b1 - learning_rate * grads['b1']  
#             self.params['b1'] += v_b1iteration %d / %d: loss %f' % (it, num_iters, loss) 
         
#             # 每一个epoch, 检查 train and val 准确率 然后将学习率衰减
#             #每次取X_batch个数据，每取够X_train的数量为一个epoch，然后将学习率衰减
#             if verbose and it % iterations_per_epoch == 0:    
                   
#                 epoch = it / iterations_per_epoch    
#                 train_acc = (self.predict(X_batch) == y_batch).mean()    
#                 val_acc = (self.predict(X_val) == y_val).mean()    
#                 train_acc_history.append(train_acc)    
#                 val_acc_history.append(val_acc)    
#                 print ('epoch %d / %d: loss %f, train_acc: %f, val_acc: %f' % 
#                                     (epoch, num_epochs, loss, train_acc, val_acc) )   
                   
#                 learning_rate *= learning_rate_decay    
#                 #mu也要变化   
#                 mu *= mu_increase

#         return {   
#             'loss_history': loss_history,   
#             'train_acc_history': train_acc_history,   
#             'val_acc_history': val_acc_history,
#         }
  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    y_pred = np.argmax(self.loss(X),axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


