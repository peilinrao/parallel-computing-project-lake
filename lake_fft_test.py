# FFT test code

import numpy as np
import time
from mpi4py import MPI

def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)
    
# vectorization version of calculating FFT
def fft_vector(x):
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError("Size Error")
        
    # most single possible fft
    n = np.arange(2)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / 2)
    X = np.dot(M, x.reshape((2, -1)))
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        K = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + K * X_odd,
                    X_even - K * X_odd])
    return X.ravel()


def fft_v_splits(x, splits):
    x = np.array([0, 1, 2, 3])
    x1 = np.array([0, 2])
    x2 = np.array([1, 3])

    # core 1 calculate X1
    X1 = fft_v(x1)

    # core 2 calculate X2
    X2 = fft_v(x2)

    # combined them
    N = x.shape[0]
    terms = np.exp(-2j * np.pi * np.arange(x.shape[0]) / x.shape[0])
    combined = np.concatenate([X1 + terms[:int(N/2)] * X2,
                                X1 + terms[int(N/2):] * X2])

# x = np.random.random(2048)


start = time.time()
print(fft_v(x))
end = time.time()
print("time needed:", end - start)
print(combined)
print(combined == fft_v(x))


# start = time.time()
# print(dft(x))
# end = time.time()
# print("time needed:", end - start)

# start = time.time()
# print(fft(x))
# end = time.time()
# print("time needed:", end - start)

