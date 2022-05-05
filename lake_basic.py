import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import math
import cmath
import time
from mpl_toolkits.mplot3d import Axes3D
import mpi4py
import sys
from mpi4py import MPI
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# set of good looking parameters
# (1) good and peace
# N = M = Lx = Lz = 256
# mu, sigma = 0, 0.001 # mean and standard deviation
# Vw = 20 # wind velocity
# Dw = (4, 4) # wind direction
# frame = 5
# A = 0.02 # magnitude

# hyperparameters
# (2) a little wave
N = M = Lx = Lz = 256
mu, sigma = 0, 1 # mean and standard deviation
Vw = 30 # wind velocity
Dw = (4, 0) # wind direction
frame = 20
A = 0.003**2 * 0.02 # magnitude

# constants
epsilon = 0
g = 9.8 # sorry newton


# (x,z) is 2d coordinates and y is height
x = np.arange(N) - int(N/2)
z = np.arange(N) - int(N/2)
x_mesh, z_mesh = np.meshgrid(x, z)
yarray = np.zeros((Lx, Lz, frame))

# cuda helper functions +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@cuda.jit(device=True)
def h0_cuda(k, conjugate, random):
    epsilon_real = random[cuda.grid(1)]
    epsilon_imag = random[cuda.grid(1) + N]

    # epsilon_real = 0.05
    # epsilon_imag = 0.05

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
def h_cuda(k, t,  random_0, random_1):
    k_length = cmath.sqrt(k[0]**2 + k[1]**2)
    w = cmath.sqrt(g*k_length)
    e_iwkt = cmath.exp(w * t * 1j)
    e_neg_iwkt = cmath.exp(-w * t * 1j)
    neg_k = (-k[0], -k[1])
    h = h0_cuda(k, False, random_0) * e_iwkt + h0_cuda(neg_k, True, random_1) * e_neg_iwkt
    return h

# rewrite h part using cuda
@cuda.jit
def h_kernal(array, t, random_0, random_1):
    pos = cuda.grid(1)
    X_index = pos / N
    Z_index = pos % N
    X_value = X_index - int(N/2)
    Z_value = Z_index - int(N/2)
    k = (2 * math.pi * X_value / Lx, 2 * math.pi *  Z_value / Lz)
    array[pos] = h_cuda(k, t,random_0, random_1)

# mode helper functions +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# input x, return X
def dft(x):
    length = int(x.shape[0])
    n = np.arange(length)
    k = z.reshape((length, 1)) # k and z are the same here, so we just use z
    M = np.exp(-2j * np.pi * k * n / length)
    return np.dot(M, x)

# combine two fft results into a new one
def fft_combine(res):
    length = res.shape[0]
    fft1 = res[:int(length/2)]
    fft2 = res[int(length/2):]

    K = np.exp(-2j * np.pi * np.arange(length) / length)
    K_split1 = K[:int(length/2)]
    K_split2 = K[int(length/2):]
    return np.concatenate([fft1 +  K_split1 * fft2,
                            fft1 +  K_split2 * fft2])

def workerID_bin2dec(workerID):
    workerID_dec = 0
    factor = 1
    for i in workerID[::-1]:
        workerID_dec += i * factor
        factor *= 2
    return workerID_dec

# vectorization version of calculating FFT
def fft_vector(x):
    length = x.shape[0]
    if np.log2(length) % 1 > 0:
        raise ValueError("Size Error")
        
    n = np.arange(2)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / 2)
    X = np.dot(M, x.reshape((2, -1)))
    while X.shape[0] < length:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        K = np.exp(-1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]
        X = np.vstack([X_even + K * X_odd, X_even - K * X_odd])
    return X.ravel()

def fft_recursion(x):
    x = np.asarray(x, dtype=np.complex128)
    length = x.shape[0]
    if np.log2(length) % 1 > 0:
        raise ValueError("Size Error")
    elif length <= 2:
        n = np.arange(length)
        k = n.reshape((length, 1))
        M = np.exp(-2j * np.pi * k * n / length)
        return np.dot(M, x)
    else:
        X_even = fft_recursion(x[::2])
        X_odd = fft_recursion(x[1::2])
        K = np.exp(-2j * np.pi * np.arange(length) / length)
        K_split1 = K[:int(length/2)]
        K_split2 = K[int(length/2):]
        return np.concatenate([X_even + K_split1 * X_odd,
                               X_even + K_split2 * X_odd])

def parallel_FFT(x, myrank, num_of_workers):
    # it describe what should each work do to perform a parallel_fft

    # test output
    current_stage = 0
    max_stage = np.log2(num_of_workers)
    myrank_bin = "{0:b}".format(myrank)

    workerID = []
    for i in myrank_bin:
        workerID.append(int(i))
    while len(workerID) < max_stage:
        workerID = [0] + workerID

        # init results
    results = x
    for i in workerID:
        results = results[:, i::2]
    results = np.apply_along_axis(fft_recursion, 1, results)

    #figure out what stage it is and what should I do
    while current_stage < max_stage:
        lastbit = workerID[-1] # 0 means odd, 1 means even
        
        if lastbit == 0:
            workerID_dec = workerID_bin2dec(workerID)
            source = (workerID_dec + 1) * (2 ** current_stage)
            tag = myrank * 1000 + source # tag == dest * 1000 + source
            recv = comm.recv(source=source, tag=tag)
            results = np.concatenate((results,recv),axis=1)
            results = np.apply_along_axis(fft_combine, 1, results)
            workerID = workerID[:-1]
            current_stage += 1
            
        else:
            workerID_dec = workerID_bin2dec(workerID)
            dest = (workerID_dec - 1) * (2 ** current_stage)
            tag = dest * 1000 + myrank # tag == dest * 1000 + source
            comm.send(results, dest=dest, tag=tag)
            return None
    return results

def h0(k, conjugate = False):
    samples = np.random.normal(mu, sigma, 2)
    epsilon_real = samples[0]
    epsilon_imag = samples[1]

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

def make_term(m_or_n, x_or_z):
    return np.exp(2j * np.pi * (m_or_n + N/2)/N * x_or_z)

# helper functions ends +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# brute force algorithm for lake algorithm
def bf(X, Z, t):
    update_y = np.zeros((Lx+1, Lz+1))
    for x_index in range(X.shape[0]):
        for z_index in  range(Z.shape[0]):
            x_value = X[x_index]
            z_value = Z[z_index]
            res_H = 0
            v = (x_value * Lx / N, z_value * Lz / M)
            
            for k_x_index in range(X.shape[0]):
                for k_z_index in  range(Z.shape[0]):
                    k_x = X[k_x_index]
                    k_z = Z[k_z_index]

                    k = (2 * math.pi * k_x / Lx, 2 * math.pi * k_z / Lz)
                    k_dot_v = k[0]*v[0] + k[1]*v[1]
                    e_ikx = complex(math.cos(k_dot_v), math.sin(k_dot_v))
                    
                    res_H += (h(k, t) * e_ikx)
            update_y[x_index, z_index] = res_H.real
    return update_y
    
# changing the algorithm to the form of FFT
def bf_vector(X, Z, t):
    h_hat = np.zeros((N+1, N+1),dtype=np.complex_)
    for x_index in range(X.shape[0]):
        for z_index in range(Z.shape[0]):
            k = (2 * np.pi * X[x_index] / Lx, 2 * np.pi *  Z[z_index] / Lz)
            h_hat[x_index][z_index] = h(k, t)

    update_y = np.zeros((Lx+1, Lz+1))
    for x_index in range(X.shape[0]):
        for z_index in  range(Z.shape[0]):
            factor = (-1)**(int(X[x_index]) + int(Z[z_index]))
            x_value = X[x_index]
            z_value = Z[z_index]
            res_H = 0
            for n in range(N):
                k_x = X[n]
                e_2pinxiN = np.exp(2j * np.pi * (k_x + N/2)/N * x_value)

                subsum = 0
                for m in range(M):
                    k_z = Z[m]
                    e_2pimziN = np.exp(2j * np.pi * (k_z + N/2)/N * z_value)

                    k = (2 * np.pi * k_x / Lx, 2 * np.pi *  k_z / Lz)
                    h_hat = h(k, t)
                    subsum += e_2pimziN * h_hat

                res_H += e_2pinxiN * subsum
            res_H *= factor
            update_y[x_index, z_index] = abs(res_H)
            
    return update_y

# changing the algorithm to the form of FFT
def bf_vector_precalch(X, Z, t):
    h_hat = np.zeros((N+1, N+1),dtype=np.complex128)
    for x_index in range(X.shape[0]):
        for z_index in range(Z.shape[0]):
            k = (2 * np.pi * X[x_index] / Lx, 2 * np.pi *  Z[z_index] / Lz)
            h_hat[x_index][z_index] = h(k, t).real

    update_y = np.zeros((N+1, N+1))
    for x_index in range(X.shape[0]):
        for z_index in range(Z.shape[0]):
            x_value = X[x_index]
            z_value = Z[z_index]
            z_hat = make_term(Z.reshape((N+1, 1)), z_value)
            x_hat = make_term(X, x_value)
            # shape check
            factor = (-1) ** (int(X[x_index]) + int(Z[z_index]))
            res_H = 0
            for n in range(N):
                subsum = 0
                for m in range(M):
                    subsum += z_hat[m] * h_hat[n][m]
                res_H += x_hat[n] * subsum
            update_y[x_index][z_index] =  res_H.real*factor
    return update_y

# changing the algorithm to the form of FFT
def bf_vector_precalch_dot(X, Z, t):
    h_hat = np.zeros((N, N),dtype=np.complex128)

    # this part can be parallelize
    for x_index in range(X.shape[0]):
        for z_index in range(Z.shape[0]):
            k = (2 * np.pi * X[x_index] / Lx, 2 * np.pi *  Z[z_index] / Lz)
            h_hat[x_index][z_index] = h(k, t).real # question: what difference?

    # this part can be make faster
    update_y = np.zeros((N, N))
    for x_index in range(X.shape[0]):
        for z_index in range(Z.shape[0]):
            x_value = X[x_index]
            z_value = Z[z_index]
            z_hat = make_term(Z.reshape((N, 1)), z_value)
            x_hat = make_term(X, x_value)
            factor = (-1) ** (int(X[x_index]) + int(Z[z_index]))
            update_y[x_index][z_index] = np.dot(x_hat, np.dot(h_hat, z_hat)).real*factor
    return update_y

def DFT(X, Z, t):
    h_hat = np.zeros((N, N),dtype=np.complex128)
    for x_index in range(X.shape[0]):
        for z_index in range(Z.shape[0]):
            k = (2 * np.pi * X[x_index] / Lx, 2 * np.pi *  Z[z_index] / Lz)
            h_hat[x_index][z_index] = h(k, t)

    factor = np.zeros((N, N))
    for x_index in range(X.shape[0]):
        for z_index in range(Z.shape[0]):
            factor[x_index][z_index] = (-1) ** (int(X[x_index]) + int(Z[z_index]))

   # this part can be make faster
    update_y = np.zeros((N, N))
    # for z_index in range(Z.shape[0]):
    x_value = X[x_index]
    z_value = Z[z_index]
    N_array = np.arange(N)
    M_array = np.arange(N).reshape((N, 1))

    # z_hat = np.exp(2j * np.pi * M_array/N * z_value)
    # firstFFT = np.dot(h_hat, z_hat)
    firstFFT = np.apply_along_axis(dft, 1, h_hat)
    firstFFT = firstFFT.T
    secondFFT = np.apply_along_axis(dft, 1, firstFFT)
    update_y = secondFFT.T.real

    return np.multiply(update_y, factor)

# Singe core without MPI, without Cuda
def FFT(X, Z, t):
    h_hat = np.zeros((N, N),dtype=np.complex128)
    for x_index in range(X.shape[0]):
        for z_index in range(Z.shape[0]):
            k = (2 * np.pi * X[x_index] / Lx, 2 * np.pi *  Z[z_index] / Lz)
            h_hat[x_index][z_index] = h(k, t)

    factor = np.zeros((N, N))
    for x_index in range(X.shape[0]):
        for z_index in range(Z.shape[0]):
            factor[x_index][z_index] = (-1) ** (int(X[x_index]) + int(Z[z_index]))

   # this part can be make faster
    update_y = np.zeros((N, N))
    x_value = X[x_index]
    z_value = Z[z_index]
    N_array = np.arange(N)
    M_array = np.arange(N).reshape((N, 1))

    # z_hat = np.exp(2j * np.pi * M_array/N * z_value)
    # firstFFT = np.dot(h_hat, z_hat)
    firstFFT = np.apply_along_axis(fft_recursion, 1, h_hat)
    firstFFT = firstFFT.T
    secondFFT = np.apply_along_axis(fft_recursion, 1, firstFFT)
    update_y = secondFFT.T.real

    return np.multiply(update_y, factor)

# Multi-core with MPI, without Cuda
def FFT_P(X, Z, t, rank, num_of_workers):
    h_hat = np.zeros((N, N),dtype=np.complex128)
    if rank == 0:
        # do this with cuda
        for x_index in range(X.shape[0]):
            for z_index in range(Z.shape[0]):
                k = (2 * np.pi * X[x_index] / Lx, 2 * np.pi *  Z[z_index] / Lz)
                h_hat[x_index][z_index] = h(k, t)
    h_hat = comm.bcast(h_hat, root=0)
    
    factor = np.zeros((N, N))
    for x_index in range(X.shape[0]):
        for z_index in range(Z.shape[0]):
            factor[x_index][z_index] = (-1) ** (int(X[x_index]) + int(Z[z_index]))

   # this part can be make faster
    update_y = np.zeros((N, N))
    # firstFFT = np.apply_along_axis(parallel_FFT, 1, h_hat, rank, num_of_workers)
    firstFFT = parallel_FFT(h_hat, rank, num_of_workers)
    firstFFT = comm.bcast(firstFFT, root=0)
    firstFFT = firstFFT.T

    # distribute this results to others
    secondFFT = parallel_FFT(firstFFT, rank, num_of_workers)
    # secondFFT = np.apply_along_axis(parallel_FFT, 1, firstFFT, rank, num_of_workers)
    if rank == 0:
        update_y = secondFFT.T.real
        result = np.multiply(update_y, factor)
        return result
    else:
        return None

# Single core without MPI, with Cuda
def FFT_C(X, Z, t):
    h_hat = np.zeros((N, N),dtype=np.complex128)
    # xoroshiro128p_normal_float32 is problematic, use precalculated random
    random_0 = np.random.normal(mu, sigma, N * N * 2)
    random_1 = np.random.normal(mu, sigma, N * N * 2)
    h_hat = np.zeros(N * N,dtype=np.complex128)
    h_kernal[N, N](h_hat, t, random_0, random_1)
    h_hat = h_hat.reshape((N, N))

    factor = np.zeros((N, N))
    for x_index in range(X.shape[0]):
        for z_index in range(Z.shape[0]):
            factor[x_index][z_index] = (-1) ** (int(X[x_index]) + int(Z[z_index]))

   # this part can be make faster
    update_y = np.zeros((N, N))
    x_value = X[x_index]
    z_value = Z[z_index]
    N_array = np.arange(N)
    M_array = np.arange(N).reshape((N, 1))

    # z_hat = np.exp(2j * np.pi * M_array/N * z_value)
    # firstFFT = np.dot(h_hat, z_hat)
    firstFFT = np.apply_along_axis(fft_recursion, 1, h_hat)
    firstFFT = firstFFT.T
    secondFFT = np.apply_along_axis(fft_recursion, 1, firstFFT)
    update_y = secondFFT.T.real

    return np.multiply(update_y, factor)

# Multi-core with MPI, with Cuda
def FFT_PC(X, Z, t, rank, num_of_workers):
    h_hat = np.zeros((N, N),dtype=np.complex128)
    # xoroshiro128p_normal_float32 is problematic, use precalculated random
    random_0 = np.random.normal(mu, sigma, N * N * 2)
    random_1 = np.random.normal(mu, sigma, N * N * 2)
    if rank == 0:
        h_hat = np.zeros(N * N,dtype=np.complex128)
        h_kernal[N, N](h_hat, t, random_0, random_1)
        h_hat = h_hat.reshape((N, N))

    h_hat = comm.bcast(h_hat, root=0)
        
    
    # factor array is simple so every worker should calculate their own one
    factor = np.zeros((N, N))
    for x_index in range(X.shape[0]):
        for z_index in range(Z.shape[0]):
            factor[x_index][z_index] = (-1) ** (int(X[x_index]) + int(Z[z_index]))

   # this part can be make faster
    update_y = np.zeros((N, N))
    # firstFFT = np.apply_along_axis(parallel_FFT, 1, h_hat, rank, num_of_workers)
    firstFFT = parallel_FFT(h_hat, rank, num_of_workers)
    firstFFT = comm.bcast(firstFFT, root=0)
    firstFFT = firstFFT.T

    # distribute this results to others
    secondFFT = parallel_FFT(firstFFT, rank, num_of_workers)
    # secondFFT = np.apply_along_axis(parallel_FFT, 1, firstFFT, rank, num_of_workers)
    if rank == 0:
        update_y = secondFFT.T.real
        result = np.multiply(update_y, factor)
        return result
    else:
        return None


if __name__ == '__main__':
    args = sys.argv[1:]
    method = args[0]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # compute and draw
    start = time.time()
    for t in range(frame):
        if method == "bf":
            yarray[:,:,t] = bf(x, z, t)
        if method == "DFT":
            yarray[:,:,t] = DFT(x, z, t)
        if method == "FFT":
            yarray[:,:,t] = FFT(x, z, t)
        if method == "FFT_P":
            if comm.size == 1:
                # reduced to single score FFT
                yarray[:,:,t] = FFT(x, z, t)
            else:
                yarray[:,:,t] = FFT_P(x, z, t, rank, comm.size)
        if method == "FFT_PC":
            if comm.size == 1:
                # reduced to single score FFT
                yarray[:,:,t] = FFT_C(x, z, t)
            else:
                yarray[:,:,t] = FFT_PC(x, z, t, rank, comm.size)
    end = time.time()

    if rank == 0:
        print("computation time needed:", end - start)

        # anime
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111, projection='3d')
        x_mesh = x_mesh.flatten()
        z_mesh = z_mesh.flatten()
        def update_plot(frame_number, yarray, plot):
            plot[0].remove()
            plot[0] = ax.plot_trisurf(x_mesh, z_mesh, yarray[:,:,frame_number].flatten(), linewidth=0.2, antialiased=True, color=(0,0,0,0), edgecolor='Blue')
        plot = [ax.plot_trisurf(x_mesh, z_mesh, yarray[:,:,0].flatten(), linewidth=0.2, antialiased=True, color=(0,0,0,0), edgecolor='Blue')]
        ax.set_zlim(-25, 25)
        ani = animation.FuncAnimation(fig, update_plot, frame, fargs=(yarray, plot), interval=1)
        
        # save to local
        # mywriter = animation.PillowWriter(fps=5)
        # ani.save('temp.gif',writer=mywriter)

        # show
        # plt.show()