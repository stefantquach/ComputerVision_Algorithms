### EDF --- An Autograd Engine for instruction
## (based on joint discussions with David McAllester)

import numpy as np
from scipy.signal import convolve2d as conv2d

# Global list of different kinds of components
ops = []
params = []
values = []


# Global forward
def Forward():
    for c in ops: c.forward()

# Global backward    
def Backward(loss):
    for c in ops:
        c.grad = np.zeros_like(c.top)
    for c in params:
        c.grad = np.zeros_like(c.top)

    loss.grad = np.ones_like(loss.top)
    for c in ops[::-1]: c.backward() 

# SGD
def SGD(lr):
    for p in params:
        p.top = p.top - lr*p.grad


## Fill this out        
def init_momentum():
    for p in params:
        p.last_grad = 0


## Fill this out
def momentum(lr,mom=0.9):
    for p in params:
        p.top = p.top - lr*(p.grad + mom*p.last_grad)
        p.last_grad = p.grad


###################### Different kinds of nodes

# Values (Inputs)
class Value:
    def __init__(self):
        values.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()

# Parameters (Weights we want to learn)
class Param:
    def __init__(self):
        params.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()


### Operations

# Add layer (x + y) where y is same shape as x or is 1-D
class add:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = self.x.top + self.y.top

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad

        if self.y in ops or self.y in params:
            if len(self.y.top.shape) < len(self.grad.shape):
                ygrad = np.sum(self.grad,axis=tuple(range(len(self.grad.shape)-1)))
            else:
                ygrad= self.grad
            self.y.grad = self.y.grad + ygrad

# Matrix multiply (fully-connected layer)
class matmul:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = np.matmul(self.x.top,self.y.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.matmul(self.y.top,self.grad.T).T
        if self.y in ops or self.y in params:
            self.y.grad = self.y.grad + np.matmul(self.x.top.T,self.grad)


# Rectified Linear Unit Activation            
class RELU:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.maximum(self.x.top,0)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad * (self.top > 0)


# Reduce to mean
class mean:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.mean(self.x.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad*np.ones_like(self.x.top) / np.float32(np.prod(self.x.top.shape))



# Soft-max + Loss (per-row / training example)
class smaxloss:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        y = self.x.top
        y = y - np.amax(y,axis=1,keepdims=True)
        yE = np.exp(y)
        yS = np.sum(yE,axis=1,keepdims=True)
        y = y - np.log(yS); yE = yE / yS

        truey = np.int64(self.y.top)
        self.top = -y[range(len(truey)),truey]
        self.save = yE

    def backward(self):
        if self.x in ops or self.x in params:
            truey = np.int64(self.y.top)
            self.save[range(len(truey)),truey] = self.save[range(len(truey)),truey] - 1.
            self.x.grad = self.x.grad + np.expand_dims(self.grad,-1)*self.save
        # No backprop to labels!    

# Compute accuracy (for display, not differentiable)        
class accuracy:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        truey = np.int64(self.y.top)
        self.top = np.float32(np.argmax(self.x.top,axis=1)==truey)

    def backward(self):
        pass


# Downsample by 2    
class down2:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = self.x.top[:,::2,::2,:]

    def backward(self):
        if self.x in ops or self.x in params:
            grd = np.zeros_like(self.x.top)
            grd[:,::2,::2,:] = self.grad
            self.x.grad = self.x.grad + grd


# Flatten (conv to fc)
class flatten:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = np.reshape(self.x.top,[self.x.top.shape[0],-1])

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.reshape(self.grad,self.x.top.shape)



# Convolution Layer
## Fill this out
class conv2:

    def __init__(self,x,k):
        ops.append(self)
        self.x = x
        self.k = k

    def forward(self):
        batch, Hx, Wx, channels = self.x.top.shape
        Hk, Wk, _, num_filters  = self.k.top.shape
        output = np.zeros((batch,Hx-Hk+1, Wx-Wk+1, num_filters))

        # iterating over spatial elements
        for j in range(Hx-Hk+1):
            for i in range(Wx-Wk+1):
                # Slicing and adding an extra dimension for broadcasting
                x_slice = np.stack([self.x.top[:,j:j+Hk, i:i+Wk,:]], axis=-1)
                output[:,j,i,:] = np.sum(x_slice * self.k.top, axis=(1,2,3))
                
        self.top = output


    def backward(self):
        batch, Hx, Wx, channels = self.x.top.shape
        Hk, Wk, _, num_filters  = self.k.top.shape
        
        if self.x in ops or self.x in params:
            xgrad = np.zeros(self.x.top.shape)
            padded = np.pad(self.grad, ((0,0),(Hk-1, Hk-1), (Wk-1, Wk-1), (0,0)))
            flipped_k = self.k.top[::-1,::-1,:,:]

            for j in range(Hx):
                for i in range(Wx):
                    grad_slice = np.stack([padded[:,j:j+Hk, i:i+Wk,:]], axis=-1)
                    flip_k_reorder = np.moveaxis(flipped_k, -1, -2)
                    xgrad[:,j,i,:] = np.sum(grad_slice*flip_k_reorder, axis=(1,2,3))

            self.x.grad = self.x.grad + xgrad
        
        if self.k in ops or self.k in params:
            _, Hy, Wy, _ = self.grad.shape
            kgrad = np.zeros(self.k.top.shape)

            for j in range(Hk): 
                for i in range(Wk):
                    x_slice = np.stack([self.x.top[:,j:j+Hy,i:i+Wy,:]], axis=-1)
                    grad_reorder = np.moveaxis(np.stack([self.grad], axis=-1), -1,-2)
                    res = x_slice * grad_reorder
                    kgrad[j,i,:,:] = np.sum(res, axis=(0,1,2))

            self.k.grad = self.k.grad + kgrad

### Extra credit 

def im2col(A, BSZ, stepsize=1, xaxis=(0,1), kaxis=(0,1)):
    # Parameters
    M,N = A.shape[xaxis[0]], A.shape[xaxis[1]]
    Hk, Wk = BSZ[kaxis[0]], BSZ[kaxis[1]]
    col_extent = (N - Wk)//stepsize + 1
    row_extent = (M - Hk)//stepsize + 1
    
    # Get Starting block indices
    start_idx = np.arange(Hk)[:,None]*N + np.arange(Wk)
    
    # Get offsetted indices across the height and width of input array
    offset_idx = (np.arange(row_extent)[:,None]*N + np.arange(col_extent))*stepsize
    
    # Get all actual indices & index into input array for final output
    flat_shape = np.array(A.shape)
    flat_shape[xaxis[0]] = M*N
    flat_shape = np.delete(flat_shape, xaxis[1])
    A_reshape = A.reshape(flat_shape)
    
    return np.take(A_reshape, start_idx.ravel()[:,None] + offset_idx.ravel(), axis=xaxis[0])

def forward_conv(x, k, stride=1):
    batch, Hx, Wx, channels = x.shape
    Hk, Wk, _, num_filters  = k.shape
    outH = (Hx-Hk)//(stride)+1
    outW = (Wx-Wk)//(stride)+1

    # Flattening kernels
    k = k.reshape(Hk*Wk, channels, num_filters)
    k = k.T.reshape(num_filters, Hk*Wk*channels)
    # Reshaping input for convolution by matrix multiply
    a_ = im2col(x, (Hk,Wk), stepsize=stride, xaxis=(1,2), kaxis=(0,1))
    a_ = np.moveaxis(a_, -1, 1)
    a_ = a_.reshape(batch, Hk*Wk*channels, outH*outW)
    # Convolution through matrix multiply
    conv = k@a_
    conv = conv.reshape(batch, outH,outW,num_filters)
    return conv

class conv_down2:
    def __init__(self,x,k, stride=1):
        ops.append(self)
        self.x = x
        self.k = k
        self.stride = stride

    def forward(self):
        # batch, Hx, Wx, channels = self.x.top.shape
        # Hk, Wk, _, num_filters  = self.k.top.shape
        # print(self.x.top.shape, self.k.top.shape)
        # outH = (Hx-Hk)//(self.stride)+1
        # outW = (Wx-Wk)//(self.stride)+1
        # output = np.zeros((batch, outH, outW, num_filters))

        # # iterating over spatial elements
        # for j in range(outH):
        #     for i in range(outW):
        #         # Slicing and adding an extra dimension for broadcasting
        #         llx = self.stride*i
        #         lly = self.stride*j 
        #         x_slice = np.stack([self.x.top[:,lly:lly+Hk, llx:llx+Wk,:]], axis=-1)
        #         output[:,j,i,:] = np.sum(x_slice * self.k.top, axis=(1,2,3))

        self.top = forward_conv(self.x.top, self.k.top, stride=self.stride)
        # print(self.top.shape)
        
        # batch, Hx, Wx, channels = self.x.top.shape
        # Hk, Wk, _, num_filters  = self.k.top.shape
        # outH = (Hx-Hk)//(self.stride)+1
        # outW = (Wx-Wk)//(self.stride)+1

        # # Flattening kernels
        # k = self.k.top.reshape(Hk*Wk, channels, num_filters)
        # k = k.T.reshape(num_filters, Hk*Wk*channels)
        # # Reshaping input for convolution by matrix multiply
        # a_ = im2col(self.x.top, (Hk,Wk), stepsize=self.stride, xaxis=(1,2), kaxis=(0,1))
        # a_ = np.moveaxis(a_, -1, 1)
        # a_ = a_.reshape(batch, Hk*Wk*channels, outH*outW)
        # # Convolution through matrix multiply
        # conv = k@a_
        # conv = conv.transpose(0,2,1).reshape(batch, outH,outW,num_filters)
        # self.top = conv



    def backward(self):
        batch, Hx, Wx, channels = self.x.top.shape
        Hk, Wk, _, num_filters  = self.k.top.shape
        batch, Hg, Wg, _  = self.grad.shape
        # print(self.grad.shape)

        # Dilating gradient
        dilated_W = 1+self.stride*(Wg-1) # same as H+(s-1)(H-1)
        dilated_H = 1+self.stride*(Hg-1)
        dilated_grad = np.zeros((batch, dilated_H, dilated_W, num_filters))
        dilated_grad[:,::self.stride,::self.stride,:] = self.grad

        if self.x in ops or self.x in params:
            # # For output
            xgrad = np.zeros(self.x.top.shape)
            # Padding dilated grad
            padded = np.pad(dilated_grad, ((0,0),(Hk-1, Hk-1), (Wk-1, Wk-1), (0,0)))
            # Flipping kernel
            flipped_k = self.k.top[::-1,::-1,:,:]

            for j in range(Hx):
                for i in range(Wx):
                    grad_slice = np.stack([padded[:,j:j+Hk, i:i+Wk,:]], axis=-1)
                    flip_k_reorder = np.moveaxis(flipped_k, -1, -2)
                    xgrad[:,j,i,:] = np.sum(grad_slice*flip_k_reorder, axis=(1,2,3))

            # self.x.grad = self.x.grad + xgrad
            # flipped_k = np.moveaxis(flipped_k, -2,-1)
            # res = forward_conv(padded, flipped_k, stride=1)
            # exit(1)
            # self.x.grad += res

        
        if self.k in ops or self.k in params:
            # kgrad = np.zeros(self.k.top.shape)
            # for j in range(Hk): 
            #     for i in range(Wk):
            #         x_slice = np.stack([self.x.top[:,j:j+dilated_H,i:i+dilated_W,:]], axis=-1)
            #         grad_reorder = np.moveaxis(np.stack([dilated_grad], axis=-1), -1,-2)
            #         res = x_slice * grad_reorder
            #         kgrad[j,i,:,:] = np.sum(res, axis=(0,1,2))

            # self.k.grad = self.k.grad + kgrad

            # Reshaping input for convolution by matrix multiply
            # a_ = im2col(self.x.top, (Hk,Wk), stepsize=self.stride, xaxis=(1,2), kaxis=(0,1))
            # a_ = np.moveaxis(a_, -1, 1)
            # a_ = a_.reshape(batch, Hk*Wk*channels, Hg*Wg)

            # grad_reshape = self.grad.reshape(num_filters, -1)
            # print(grad_reshape.shape)
            # exit(1)
            print(self.k.top.shape)
            print(self.grad.shape)
            x_reshape = self.x.top.transpose(3,1,2,0)
            print(x_reshape.shape)
            grad_reshape = self.grad.transpose(1,2,0,3)
            print(grad_reshape.shape)
            kgrad = forward_conv(x_reshape, grad_reshape)
            print(kgrad.shape)
            self.k.grad = self.k.grad + kgrad.transpose(1,2,0,3)

