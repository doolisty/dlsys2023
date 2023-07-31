import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    image_size = 28
    image_num = 60000
    with gzip.open(image_filesname, "r") as f:
        f.read(16)
        image_arr = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)
    with gzip.open(label_filename, "r") as f:
        f.read(8)
        label_arr = np.frombuffer(f.read(), dtype=np.uint8)
    image_arr.resize(image_num, image_size ** 2)
    image_arr -= np.min(image_arr)
    image_arr /= np.max(image_arr)
    return (image_arr, label_arr)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # print(f"Z.shape = {Z.shape}")
    # print(f"y_one_hot.shape = {y_one_hot.shape}")
    # print(y_one_hot)
    # lhs = Z.exp().sum((1)).log()
    # rhs = (Z * y_one_hot).sum((1))
    # # print(f"lhs.shape = {lhs.shape}")
    # # print(f"rhs.shape = {rhs.shape}")
    # return (lhs - rhs).sum() / Z.shape[0]
    return (Z.exp().sum((1)).log() - (Z * y_one_hot).sum((1))).sum() / Z.shape[0]
    # return np.mean(np.log(np.sum(np.exp(Z), axis=1)) - y)
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    k = W2.shape[1]
    sr = 0  # start_row
    while sr < num_examples:
        batch = min(batch, num_examples - sr)
        tensor_X = ndl.Tensor(X[sr:sr + batch])
        Iy = np.zeros((batch, k))
        Iy[np.arange(batch), y[sr:sr + batch]] = 1
        tensor_Iy = ndl.Tensor(Iy)

        tmp_Z1 = tensor_X @ W1
        Z1 = tmp_Z1.relu()
        G2 = (Z1 @ W2).exp()
        # broadcast outside sum, need to first expand inside then transpose
        def transpose_shape(shape):
            shape_lst = list(shape)
            shape_lst[-2], shape_lst[-1] = shape_lst[-1], shape_lst[-2]
            return tuple(shape_lst)
        G2 /= G2.sum(1).broadcast_to(transpose_shape(G2.shape)).transpose()
        G2 -= tensor_Iy
        G1 = Z1 / tmp_Z1
        G1 = G1 * (G2 @ W2.transpose())

        grad1 = tensor_X.transpose() @ G1 / batch
        grad2 = Z1.transpose() @ G2 / batch

        W1 -= lr * grad1
        W2 -= lr * grad2

        sr += batch
    return (W1, W2)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
