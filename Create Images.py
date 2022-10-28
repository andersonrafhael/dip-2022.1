import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# PDI 
# Student: Anderson Rafhael; 18113000

# Quiz 1: Create Images

# General Functions

def normalize(min, max, zz):
  return ((max-min) * ((zz - np.min(zz)) / (np.max(zz) - np.min(zz)))).astype(int) + min


def plotFilledContours(xx, yy, zz, tittle):
    
    zNormalized = normalize(0, 255, zz)
    
    plt.figure()
    plt.subplot()
    plt.title(tittle)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.imshow(zNormalized, cmap=cm.magma)
    plt.colorbar()
    
    """fig = plt.contourf(xx, yy, zNormalized, cmap=cm.magma)
    plt.title(tittle)
    plt.axis('scaled')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()"""
    
    plt.show()
    
    
def plot3D(xx, yy, zz, tittle):
    
    fig = plt.figure()
    ax = fig.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xx, yy, zz, cmap=cm.magma, linewidth=0, antialiased=True, cstride=2, rstride=2)
    
    plt.title(tittle)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


def gaussFucntion(x, y, mx, my, sx, sy):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


def rotatedGaussFunction(shape = (100, 100), mx = 50, my = 50, sx = 10, sy = 10, theta = 0):
    
    xx0, yy0 = np.meshgrid(range(shape[1]), range(shape[0]))
    xx0 -= mx
    yy0 -= my
    theta = np.deg2rad(theta)
    xx = xx0 * np.cos(theta) - yy0 * np.sin(theta)
    yy = xx0 * np.sin(theta) + yy0 * np.cos(theta)
    try:
        img = np.exp( - ((xx**2)/(2*sx**2) + 
                         (yy**2)/(2*sy**2)) )
    except ZeroDivisionError:
        img = np.zeros((shape[0], shape[1]), dtype='float64')
        
    plotFilledContours(xx, yy, img, "2D Rotated Gauss Rotated")
    plot3D(xx, yy, img, "3D Rotated Gauss Rotated")
    
    return img



#Question 1: Create the image of a paraboloid with one axis scaled (like an oval paraboloid).
x = np.linspace(-10, 10, 101)
y = np.linspace(-10, 10, 101)

xx, yy = np.meshgrid(x, y)

zz = xx**2 + yy**2

plotFilledContours(xx, yy, zz, "2D Paraboloid")
plot3D(xx, yy, zz, "3D Paraboloid")



#Question 2: Create the image of a rotated sin(x) using rotation of coordinates.
x = np.linspace(-10, 10, 1001)
y = np.linspace(-10, 10, 1001)

xx, yy = np.meshgrid(x, y)

zz = (np.sin(np.sqrt(xx**2 + yy**2))+1) / 2

plotFilledContours(xx, yy, zz, "2D Rotated Sin")
plot3D(xx, yy, zz, "3D Rotated Sin")



#Question 3: Create the image of a gaussian in 2D.
x = np.linspace(-5, 5, 101)
y = np.linspace(-5, 5, 101)

xx, yy = np.meshgrid(x, y)

zz = gaussFucntion(xx, yy, 0, 0, 1, 1)

plotFilledContours(xx, yy, zz, "2D Gauss")
plot3D(xx, yy, zz, "3D Gauss")



#Question 4: Create a function that generates the image of a Gaussian optionally rotated by an angle \theta and with mx, my, sx, sy as input arguments.
rotatedGaussFunction(shape = (101, 101), mx = 30, my = 40, sx = 10, sy = 10, theta = np.pi/4)