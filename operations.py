import numpy as np
from utils.tools import *

# Attention:
# - Never change the value of inputs passed into the functions,
#   which will change the result of backward


class operation(object):
    """
    Operation abstraction
    """

    def forward(self, inputs):
        """Forward operation, reture output"""
        raise NotImplementedError

    def backward(self, out_grad, inputs):
        """Backward operation, return gradient to inputs"""
        raise NotImplementedError


class relu(operation):
    def __init__(self):
        super(relu, self).__init__()

    def forward(self, inputs):
        output = np.maximum(0, inputs)
        return output

    def backward(self, out_grad, inputs):
        in_grad = (inputs >= 0) * out_grad
        return in_grad


class flatten(operation):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, inputs):
        batch = inputs.shape[0]
        output = inputs.copy().reshape(batch, -1)
        return output

    def backward(self, out_grad, inputs):
        in_grad = out_grad.copy().reshape(inputs.shape)
        return in_grad


class matmul(operation):
    def __init__(self):
        super(matmul, self).__init__()

    def forward(self, inputs, weights):
        """
        # Arguments
            inputs: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        return np.matmul(inputs, weights)

    def backward(self, out_grad, inputs, weights):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            inputs: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            in_grad: gradient to the forward inputs with same shape as inputs
            w_grad: gradient to weights, with same shape as weights            
        """
        in_grad = np.matmul(out_grad, weights.T)
        w_grad = np.matmul(inputs.T, out_grad)
        return in_grad, w_grad


class add_bias(operation):
    def __init__(self):
        super(add_bias, self).__init__()

    def forward(self, inputs, bias):
        """
        # Arugments
          inputs: numpy array with shape (batch, in_features)
          bias: numpy array with shape (in_features)

        # Returns
          output: numpy array with shape(batch, in_features)
        """
        return inputs + bias.reshape(1, -1)

    def backward(self, out_grad, inputs, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            inputs: numpy array with shape (batch, in_features)
            bias: numpy array with shape (out_features)
        # Returns
            in_grad: gradient to the forward inputs with same shape as inputs
            b_bias: gradient to bias, with same shape as bias
        """
        in_grad = out_grad
        b_grad = np.sum(out_grad, axis=0)
        return in_grad, b_gra


class RNNCellOp(operation):
    def __init__(self):
        """
        # Arguments
            in_features: int, the number of inputs features
            units: int, the number of hidden units
            initializer: Initializer class, to initialize weights
        """
        super(RNNCellOp, self).__init__()

    def forward(self, inputs, kernel, recurrent_kernel, bias):
        """
        # Arguments
            inputs: [inputs numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)]
            Refer to the main.ipynb for the other arguments.
        # Returns
            outputs: numpy array with shape (batch, units)
        """
        output = None
        ###############################################
        
        x, prev_h = inputs
        U = kernel 
        W = recurrent_kernel
        b = bias 
        a_t = np.dot(x,U) + np.dot(prev_h,W) + b
        output = np.tanh(a_t)
        
        
            
        
       
        
      
        
        ###############################################
        return output

    def backward(self, out_grad, inputs, kernel, recurrent_kernel, bias):
        """
        # Arguments
            out_grads: numpy array with shape (batch, units), gradients to outputs
            inputs: [inputs numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)], same with forward inputs

        # Returns
            in_grads: [gradients to inputs numpy array with shape (batch, in_features), 
                        gradients to state numpy array with shape (batch, units)]
            ...
        """
        in_grad = None
        kernel_grad = None
        r_kernel_grad = None
        b_grad = None
        ###############################################
    
        x, prev_h = inputs
        x = np.nan_to_num(x)
        
        U = kernel 
        W = recurrent_kernel
        b = bias 
        next_h = self.forward(inputs, U, W, b)
        
        df = out_grad * (1 - np.square(next_h))  ## dout_d(tanh(x)) /dx = 1 - tanh(x)** 2  and out_grad  is the gradient to the forward output of fc layer,which equals to the dnext_h 
        df = np.nan_to_num(df)
        
        dx = np.dot(df, U.T)       ##  dx = df.(U.T)
        dx = np.nan_to_num(dx)
        
        dprev_h = np.dot( df, W.T)     ##  dprev_h =df.(W.T)
        dprev_h = np.nan_to_num(dprev_h)
        
        in_grad = [dx,dprev_h]
  
        
        dU = np.dot(x.T,df)                ## dU = (x.T).df
        kernel_grad =np.nan_to_num(dU)
        
        dW = np.dot(prev_h.T, df)           ## dW = prev_h.T.df
        r_kernel_grad = np.nan_to_num(dW)
        
        db = np.sum(df, axis=0)
        b_grad = np.nan_to_num(db)
        
        
        
        
        
        ###############################################
        return in_grad, kernel_grad, r_kernel_grad, b_grad


class softmax_cross_entropy(operation):
    def __init__(self):
        super(softmax_cross_entropy, self).__init__()

    def forward(self, inputs, labels):
        """
        # Arguments
            inputs: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            output: scalar, average loss
            probs: the probability of each category
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        inputs_shift = inputs - np.max(inputs, axis=1, keepdims=True)
        Z = np.sum(np.exp(inputs_shift), axis=1, keepdims=True)

        log_probs = inputs_shift - np.log(Z + eps)
        probs = np.exp(log_probs)
        output = -1 * np.sum(log_probs[np.arange(batch), labels]) / batch
        return output, probs

    def backward(self, inputs, labels):
        """
        # Arguments
            inputs: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            in_grad: gradient to forward inputs of softmax cross entropy, with shape (batch, num_class)
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        inputs_shift = inputs - np.max(inputs, axis=1, keepdims=True)
        Z = np.sum(np.exp(inputs_shift), axis=1, keepdims=True)
        log_probs = inputs_shift - np.log(Z + eps)
        probs = np.exp(log_probs)

        in_grad = probs.copy()
        in_grad[np.arange(batch), labels] -= 1
        in_grad /= batch
        return in_grad
