import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import math
import cmath
from mpl_toolkits.mplot3d import Axes3D

# method
method = "bf"
# method = "fft"

# set N = M = Lx = Lz for simplicity
N = 10 # sample density for x
M = 10 # sample density for y
Lx = 10 # X range in coordinate
Lz = 10 # Z range in coordinate
N = M = Lx = Lz = 10

# constants
g = 9.8 # sorry newton
Vw =3 # wind velocity
Dw = (-4.5, 9.4) # wind direction
frame = 40
A = 3 # magnitude

# (x,z) is 2d coordinates and y is height
x = np.arange(-Lx/2,Lx/2+1,1)
x[int(Lx/2)] = 0.000001
z = np.arange(-Lz/2,Lz/2+1,1)
z[int(Lz/2)] = 0.000001
x_mesh, z_mesh = np.meshgrid(x, z)
yarray = np.zeros((Lx+1, Lz+1, frame))

mu, sigma = 0, 1 # mean and standard deviation
samples = np.random.normal(mu, sigma, 2)
epsilon_real = samples[0]
epsilon_imag = samples[1]

# this section denotes the the brute force algorithm
def h0(k, conjugate = False):
    k_length = math.sqrt(k[0]**2 + k[1]**2)
    k_unit = (k[0]/k_length, k[1]/k_length)
    Dw_length = math.sqrt(Dw[0]**2 + Dw[1]**2)
    Dw_unit = (Dw[0]/Dw_length, Dw[1]/Dw_length)
    L = Vw**2/g
    Ph = A * math.e**(-1/((k_length*L)**2)) / k_length**4 * (k_unit[0] * Dw_unit[0] + k_unit[1] * Dw_unit[1])**2

    if conjugate:
        return complex(1/math.sqrt(2) * epsilon_real * Ph, -1/math.sqrt(2) * epsilon_imag * Ph)
    return complex(1/math.sqrt(2) * epsilon_real * Ph, 1/math.sqrt(2) * epsilon_imag * Ph)

def H(X, Z, t):
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

                    #(k_x, k_z) is just a point in the plane

                    k = (2 * math.pi * k_x / Lx, 2 * math.pi * k_z / Lz)
                    k_dot_v = k[0]*v[0] + k[1]*v[1]
                    e_ikx = complex(math.cos(k_dot_v), math.sin(k_dot_v))
                    
                    k_length = math.sqrt(k[0]**2 + k[1]**2)
                    w = math.sqrt(g*k_length)
                    e_iwkt = complex(math.cos(w*t), math.sin(w*t))
                    e_neg_iwkt = complex(math.cos(-w*t), math.sin(-w*t))

                    neg_k = (-k[0], -k[1])
                    h = h0(k, False) * e_iwkt + h0(neg_k, True) * e_neg_iwkt
                    res_H += (h * e_ikx)
            update_y[x_index, z_index] = res_H.real
            # print(res_H)
    print("done for t = ", t)
    print("yarray is", update_y)
    return update_y
    
       
def FFT(X, Z, t):
    res_H = 0
    # for n in range(N):
    #     for m in range(M):

    pass

# all computation is done here
for t in range(frame):
    if method == "bf":
        yarray[:,:,t] = H(x, z, t/10)
    if method == "fft":
        yarray[:,:,t] = FFT(x, z, t/8)


fig = plt.figure(figsize=(12,12))

ax = fig.add_subplot(111, projection='3d')


def update_plot(frame_number, yarray, plot):
    plot[0].remove()
    plot[0] = ax.plot_trisurf(x_mesh.flatten(), z_mesh.flatten(), yarray[:,:,frame_number].flatten(), linewidth=0.2, antialiased=True, color='blue')

plot = [ax.plot_trisurf(x_mesh.flatten(), z_mesh.flatten(), yarray[:,:,0].flatten(), linewidth=0.2, antialiased=True, color='blue')]
ax.set_zlim(-25, 25)
ani = animation.FuncAnimation(fig, update_plot, frame, fargs=(yarray, plot), interval=200)
plt.show()