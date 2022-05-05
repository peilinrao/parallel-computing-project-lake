# FFT test code

import numpy as np
import time
from mpi4py import MPI
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
import numba
import math
import cmath

N = M = Lx = Lz = 256
mu, sigma = 0, 1 # mean and standard deviation
Vw = 30 # wind velocity
Dw = (4, 0) # wind direction
frame = 20
A = 0.02 # magnitude

# constants
epsilon = 0
g = 9.8 # sorry newton


@cuda.jit(device=True)
def h0(k, conjugate, rng):
    # epsilon_real = xoroshiro128p_normal_float32(rng, cuda.grid(1))
    # epsilon_imag = xoroshiro128p_normal_float32(rng, cuda.grid(1))

    epsilon_real = 0.05
    epsilon_imag = 0.05

    k_length = cmath.sqrt(k[0]**2 + k[1]**2)

    if k_length == 0:
        k_unit = (0, 0)
    else:
        k_unit = (k[0]/k_length, k[1]/k_length)

    Dw_length = cmath.sqrt(Dw[0]**2 + Dw[1]**2)
    Dw_unit = (Dw[0]/Dw_length, Dw[1]/Dw_length)
    L = Vw**2/g

    kDw_length = abs(k_unit[0] * Dw_unit[0] + k_unit[1] * Dw_unit[1])
    if k_length == 0:
        Ph = 0
    else:
        Ph = cmath.sqrt(A * cmath.exp(-1/((k_length*L)**2)) / (k_length**4) * (kDw_length**2))

    
    result = 1/cmath.sqrt(2) * float(epsilon_real) * Ph +  1/cmath.sqrt(2) * float(epsilon_imag) * Ph * 1j
    if conjugate:
        result = 1/cmath.sqrt(2) * float(epsilon_real) * Ph - 1/cmath.sqrt(2) * float(epsilon_imag) * Ph * 1j

    return result

@cuda.jit(device=True)
def h(k, t, rng):
    k_length = cmath.sqrt(k[0]**2 + k[1]**2)
    w = cmath.sqrt(g*k_length)
    e_iwkt = cmath.exp(w * t * 1j)
    e_neg_iwkt = cmath.exp(-w * t * 1j)
    neg_k = (-k[0], -k[1])
    h = h0(k, False, rng) * e_iwkt + h0(neg_k, True, rng) * e_neg_iwkt
    return h

# rewrite h part using cuda
@cuda.jit
def h_kernal(array, t, rng):
    pos = cuda.grid(1)
    X_index = pos / N
    Z_index = pos % N
    X_value = X_index - int(N/2)
    Z_value = Z_index - int(N/2)
    k = (2 * math.pi * X_value / Lx, 2 * math.pi *  Z_value / Lz)
    array[pos] = h(k, t, rng)


rng = create_xoroshiro128p_states(N * N, seed=1)
h_hat = np.zeros(N * N,dtype=np.complex128)
print('Initial array:', h_hat)
h_kernal[N, N](h_hat, 1, rng)
print('Result array:', h_hat)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def h0(k, conjugate = False):
    samples = np.random.normal(mu, sigma, 2)
    epsilon_real = 0.05
    epsilon_imag = 0.05

    # epsilon_real = 0.05
    # epsilon_imag = 0.05

    k_length = math.sqrt(k[0]**2 + k[1]**2)

    if k_length == 0:
        k_unit = (0, 0)
    else:
        k_unit = (k[0]/k_length, k[1]/k_length)

    Dw_length = math.sqrt(Dw[0]**2 + Dw[1]**2)
    Dw_unit = (Dw[0]/Dw_length, Dw[1]/Dw_length)
    L = Vw**2/g

    kDw_length = abs(k_unit[0] * Dw_unit[0] + k_unit[1] * Dw_unit[1])
    if k_length == 0:
        Ph = 0
    else:
        Ph = math.sqrt(A * np.exp(-1/((k_length*L)**2)) / (k_length**4) * (kDw_length**2))

    
    result = complex(1/math.sqrt(2) * epsilon_real * Ph, 1/math.sqrt(2) * epsilon_imag * Ph)
    if conjugate:
        result = complex(1/math.sqrt(2) * epsilon_real * Ph, -1/math.sqrt(2) * epsilon_imag * Ph)

    return result

def h(k, t):
    k_length = math.sqrt(k[0]**2 + k[1]**2)
    w = math.sqrt(g*k_length)
    e_iwkt = np.exp(w * t * 1j)
    e_neg_iwkt = np.exp(-w * t * 1j)
    neg_k = (-k[0], -k[1])
    h = h0(k, False) * e_iwkt + h0(neg_k, True) * e_neg_iwkt
    return h

X = np.arange(N) - int(N/2)
Z = np.arange(N) - int(N/2)
h_hat = np.zeros(N * N,dtype=np.complex128)
for x_index in range(X.shape[0]):
    for z_index in range(Z.shape[0]):
        k = (2 * np.pi * X[x_index] / Lx, 2 * np.pi *  Z[z_index] / Lz)
        h_hat[x_index * N + z_index] = h(k, 1)
print(h_hat)


