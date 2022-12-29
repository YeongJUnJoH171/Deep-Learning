import numpy as np
from skimage.util.shape import view_as_windows


#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * filter_width * filter_height / 2),
                                  (num_filters, in_ch_size, filter_width, filter_height))
        self.b = 0.01 + np.zeros((1, num_filters, 1, 1))
        self.input_size = input_size

        #######
        ## If necessary, you can define additional class variables here
        #######

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    #######
    # Q1. Complete this method
    #######
    def forward(self, x):
        N, C, W, H = x.shape
        F, C, WW, HH = self.W.shape
        pad = 0
        stride = 1
        size = C*WW*HH
        W_out = int(1 + (W-WW)/stride)
        H_out = int(1 + (H-HH)/stride)
        out = np.zeros((N, F, W_out, H_out))

        for i in range(W_out):
            for j in range(H_out):
                x_selected = x[np.arange(N), :, i*stride:i*stride+WW, j*stride:j*stride+HH]
                x_stretched = x_selected.reshape(N, size)
                w_stretched = self.W.reshape(F, size)
                small_out = np.dot(x_stretched, w_stretched.T) + self.b.reshape((1, F))
                out[np.arange(N), :, i, j] = small_out
        return out

    #######
    # Q2. Complete this method
    #######
    def backprop(self, x, dLdy):
        N, C, W, H = x.shape
        F, C, WW, HH = self.W.shape
        N, F, W_out, H_out = dLdy.shape
        pad = 0
        stride = 1
        size = C*WW*HH
        dLdx = np.zeros((N, C, W, H))
        dLdW = np.zeros((F, C, WW, HH))
        dLdb = np.zeros((1,F,1,1))

        for i in range(W_out):
            for j in range(H_out):
                out = dLdy[np.arange(N), :, i, j]

                w_reshape = self.W.reshape(F, size)
                dx = np.dot(out, w_reshape).reshape(N, C, WW, HH)
                dLdx[np.arange(N), :, i*stride:i*stride+WW, j*stride:j*stride+HH] += dx

                x_small = x[np.arange(N), :, i*stride:i*stride+WW, j*stride:j*stride+HH]
                dW = np.dot(out.T, x_small.reshape(N, size))
                dLdW += dW.reshape(F,C,WW,HH)

                dLdb += np.sum(out, axis=0).reshape(1,F,1,1)
        return dLdx, dLdW, dLdb

    #######
    ## If necessary, you can define additional class methods here
    #######


class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        #######
        ## If necessary, you can define additional class variables here
        #######

    #######
    # Q3. Complete this method
    #######
    def forward(self, x):
        N, C, W, H = x.shape
        W_out = int((W - self.pool_size) / self.stride + 1)
        H_out = int((H - self.pool_size) / self.stride + 1)
        out = np.zeros((N, C, W_out, H_out))
        for i in range(W_out):
            for j in range(H_out):
                x_selected = x[np.arange(N), :, i*self.stride:i*self.stride + self.pool_size, j*self.stride:j*self.stride + self.pool_size]
                out[np.arange(N), :, i, j] = np.max(x_selected, axis=(2,3))
        return out

    #######
    # Q4. Complete this method
    #######
    def backprop(self, x, dLdy):
        N, C, W, H = x.shape
        N, C, W_out, H_out = dLdy.shape
        pool_size = self.pool_size ** 2
        dLdx = np.zeros((N, C, W, H))
        for i in range(W_out):
            for j in range(H_out):
                dmax_val = dLdy[np.arange(N), :, i, j].reshape((N, C, 1))

                x_selected = x[np.arange(N), :, i*self.stride:i*self.stride + self.pool_size, j*self.stride:j*self.stride+self.pool_size]
                x_stretched = x_selected.reshape((N, C, pool_size))
                max_value = np.max(x_stretched, axis=2).reshape((N, C, 1))
                dx_stretched = np.where(x_stretched==max_value,dmax_val,0)
                dx = dx_stretched.reshape((N, C, self.pool_size, self.pool_size))
                dLdx[np.arange(N), :, i*self.stride:i*self.stride + self.pool_size, j*self.stride:j*self.stride+self.pool_size] += dx
        return dLdx

    #######
    ## If necessary, you can define additional class methods here
    #######


# testing the implementation

# data sizes
batch_size = 8
input_size = 32
filter_width = 3
filter_height = filter_width
in_ch_size = 3
num_filters = 8

std = 1e0
dt = 1e-3

# number of test loops
num_test = 20

# error parameters
err_dLdb = 0
err_dLdx = 0
err_dLdW = 0
err_dLdx_pool = 0

for i in range(num_test):
    # create convolutional layer object
    cnv = nn_convolutional_layer(filter_width, filter_height, input_size, in_ch_size, num_filters, std)

    x = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    # dLdx test
    print('dLdx test')
    y1 = cnv.forward(x)
    y2 = cnv.forward(x + delta)

    bp, _, _ = cnv.backprop(x, np.ones(y1.shape))

    exact_dx = np.sum(y2 - y1) / dt
    apprx_dx = np.sum(delta * bp) / dt
    print('exact change', exact_dx)
    print('apprx change', apprx_dx)

    err_dLdx += abs((apprx_dx - exact_dx) / exact_dx) / num_test * 100

    # dLdW test
    print('dLdW test')
    W, b = cnv.get_weights()
    dW = np.random.normal(0, 1, W.shape) * dt
    db = np.zeros(b.shape)

    z1 = cnv.forward(x)
    _, bpw, _ = cnv.backprop(x, np.ones(z1.shape))
    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_dW = np.sum(z2 - z1) / dt
    apprx_dW = np.sum(dW * bpw) / dt
    print('exact change', exact_dW)
    print('apprx change', apprx_dW)

    err_dLdW += abs((apprx_dW - exact_dW) / exact_dW) / num_test * 100

    # dLdb test
    print('dLdb test')

    W, b = cnv.get_weights()

    dW = np.zeros(W.shape)
    db = np.random.normal(0, 1, b.shape) * dt

    z1 = cnv.forward(x)

    V = np.random.normal(0, 1, z1.shape)

    _, _, bpb = cnv.backprop(x, V)

    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_db = np.sum(V * (z2 - z1) / dt)
    apprx_db = np.sum(db * bpb) / dt

    print('exact change', exact_db)
    print('apprx change', apprx_db)
    err_dLdb += abs((apprx_db - exact_db) / exact_db) / num_test * 100

    # max pooling test
    # parameters for max pooling
    stride = 2
    pool_size = 2

    mpl = nn_max_pooling_layer(stride=stride, pool_size=pool_size)

    x = np.arange(batch_size * in_ch_size * input_size * input_size).reshape(
        (batch_size, in_ch_size, input_size, input_size)) + 1
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    print('dLdx test for pooling')
    y1 = mpl.forward(x)
    dLdy = np.random.normal(0, 10, y1.shape)
    bpm = mpl.backprop(x, dLdy)

    y2 = mpl.forward(x + delta)

    exact_dx_pool = np.sum(dLdy * (y2 - y1)) / dt
    apprx_dx_pool = np.sum(delta * bpm) / dt
    print('exact change', exact_dx_pool)
    print('apprx change', apprx_dx_pool)

    err_dLdx_pool += abs((apprx_dx_pool - exact_dx_pool) / exact_dx_pool) / num_test * 100

# reporting accuracy results.
print('accuracy results')
print('conv layer dLdx', 100 - err_dLdx, '%')
print('conv layer dLdW', 100 - err_dLdW, '%')
print('conv layer dLdb', 100 - err_dLdb, '%')
print('maxpool layer dLdx', 100 - err_dLdx_pool, '%')