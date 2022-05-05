# FFT test code

import numpy as np
import time
from mpi4py import MPI

def dft(x):
    x = np.asarray(x, dtype=float)
    N = int(x.shape[0])
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

# combine two fft results into a new one
def fft_combine(fft1, fft2):
    N = fft1.shape[0] + fft2.shape[0]
    K = np.exp(-2j * np.pi * np.arange(N) / N)
    K_split1 = K[:int(N/2)]
    K_split2 = K[int(N/2):]
    return np.concatenate([fft1 +  K_split1 * fft2,
                            fft1 +  K_split2 * fft2])

# vectorization version of calculating FFT
def fft_vector(x):
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError("Size Error")
        
    N_min = 2
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        K = np.exp(-1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]
        X = np.vstack([X_even + K * X_odd,
                    X_even - K * X_odd])
    return X.ravel()


def workerID_bin2dec(workerID):
    workerID_dec = 0
    factor = 1
    for i in workerID[::-1]:
        workerID_dec += i * factor
        factor *= 2
    return workerID_dec

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
        results = results[i::2]
    results = fft_vector(results)

    #figure out what stage it is and what should I do
    while current_stage < max_stage:
        lastbit = workerID[-1] # 0 means odd, 1 means even
        
        if lastbit == 0:
            workerID_dec = workerID_bin2dec(workerID)
            source = (workerID_dec + 1) * (2 ** current_stage)
            tag = myrank * 1000 + source # tag == dest * 1000 + source
            recv = comm.recv(source=source, tag=tag)
            results = fft_combine(results, recv)
            workerID = workerID[:-1]
            current_stage += 1
            
        else:
            workerID_dec = workerID_bin2dec(workerID)
            dest = (workerID_dec - 1) * (2 ** current_stage)
            tag = dest * 1000 + myrank # tag == dest * 1000 + source
            comm.send(results, dest=dest, tag=tag)
            return None

    return results
    

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(rank)
    x = np.arange(8192*256) 
    x = np.asarray(x, dtype=np.complex128)  

    if comm.size == 1:
        start = time.time()
        result = fft_vector(x)
        print(result)
        print("Ground truth")
        end = time.time()
        print(result.shape)
        print("time needed:", end - start)
    else: 
        start = time.time()
        result = parallel_FFT(x, rank, comm.size)
        end = time.time()
        if rank == 0:
            print("My result ")
            print(result)
            print(result.shape)
            print("time needed:", end - start)


