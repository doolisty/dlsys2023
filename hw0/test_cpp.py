import numpy as np
import numdifftools as nd
import sys
sys.path.append("./src")
from simple_ml import *
from simple_ml_ext import *


np.random.seed(0)
X = np.random.randn(50,5).astype(np.float32)
y = np.random.randint(3, size=(50,)).astype(np.uint8)
Theta = np.zeros((5,3), dtype=np.float32)
dTheta = -nd.Gradient(lambda Th : softmax_loss(X@Th.reshape(5,3),y))(Theta)
print("before cpp")
softmax_regression_epoch_cpp(X,y,Theta,lr=1.0,batch=50)
print("after cpp")
np.testing.assert_allclose(dTheta.reshape(5,3), Theta, rtol=1e-4, atol=1e-4)


# test multi-steps on MNIST
X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                    "data/train-labels-idx1-ubyte.gz")
theta = np.zeros((X.shape[1], y.max()+1), dtype=np.float32)
softmax_regression_epoch_cpp(X[:100], y[:100], theta, lr=0.1, batch=10)
np.testing.assert_allclose(np.linalg.norm(theta), 1.0947356, 
                            rtol=1e-5, atol=1e-5)
