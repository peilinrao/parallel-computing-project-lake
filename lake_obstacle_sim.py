import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

N = 200 
frn = 50 

x = np.linspace(-4,4,N+1)
x, y = np.meshgrid(x, x)

zarray = np.zeros((N+1, N+1, frn))

def f(x,y,sig):
    return 1/np.sqrt(sig)*np.exp(-(x**2+y**2)/sig**2)

# all computation is done here
for i in range(frn):
    zarray[:,:,i] = f(x,y,1.5+np.sin(i*2*np.pi/frn))

def update_plot(frame_number, zarray, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, zarray[:,:,frame_number], cmap="magma")

fig = plt.figure()
ax = p3.Axes3D(fig)

plot = [ax.plot_surface(x, y, zarray[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_zlim(0,1.1)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(zarray, plot), interval=10)
plt.show()