#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    /*
        Z = np.exp(X[sr:sr + batch] @ theta)
        Z /= np.sum(Z, axis=1)[:, None]
        Iy = np.zeros((batch, k))
        Iy[np.arange(batch), y[sr:sr + batch]] = 1
        grad = X[sr:sr + batch].T @ (Z - Iy) / batch
        theta -= lr * grad
    */

    size_t sr = 0;  // start_row
    float *Z = new float[batch * k];  // m * k
    printf("before while\n");
    while (sr < m) {
        batch = std::min(batch, m - sr);

        printf("before loop 1\n");
        // loop 1: calc (Z - Iy)
        for (size_t i = 0; i < batch; ++i) {
            float sum_exp_row = 0.0f;
            for (size_t j = 0; j < k; ++j) {
                float tmp_sum = 0.0f;
                for (size_t idx_k = 0; idx_k < n; ++idx_k) {
                    // Z[i][j] += batch_X[i][k] * theta[k][j]
                    tmp_sum += X[(sr + i) * n + idx_k] * theta[idx_k * k + j];
                }
                float exp_val = exp(tmp_sum);
                Z[i * k + j] = exp_val;  // write Z once
                sum_exp_row += exp_val;
            }
            for (size_t j = 0; j < k; ++j) {
                Z[i * k + j] /= sum_exp_row;  // read & write Z once
                if ((int)y[sr + i] == j) {
                    --Z[i * k + j];
                }
            }
        }

        printf("before loop 2\n");
        // loop 2: calc grad (no need to store), update theta
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < k; ++j) {
                float tmp_sum = 0.0f;
                for (size_t idx_k = 0; idx_k < m; ++idx_k) {
                    // grad[i][j] += X[k][i] * (Z-Iy)[k][j]
                    tmp_sum += X[(sr + idx_k) * n + i] * Z[idx_k * k + j];
                }
                theta[i * k + j] -= lr * tmp_sum / batch;
            }
        }

        sr += batch;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
