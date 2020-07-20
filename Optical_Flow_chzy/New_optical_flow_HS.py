from scipy.signal import convolve2d
from pylab import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy as sp
import scipy.sparse
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve

#This is HS without iterations

# Function to return image and time derivatives Ix, Iy, It
def getGradients(Img1, Img2):

    n = len(Img1[0, :])  # width
    m = len(Img1[:, 0])  # height


    Gx = zeros((n,n))
    Gy = zeros((m,m))

    Gy[0, 0] = -1
    Gy[0, 1] = 1
    Gy[m-1,m-1]=1
    Gy[m - 1, m - 2] = -1
    Gx[0, 0] = -1
    Gx[1, 0] = 1
    Gx[n - 1, n - 1] = 1
    Gx[n - 2, n - 1] = -1

    for i in range(1,m-1):

          Gy[i,i-1] = -1
          Gy[i, i+1] = 1


    # the defination of Gx to compute the central difference in x direction.
    for i in range(1,n-1):

          Gx[i-1,i] = -1
          Gx[i+1, i] = 1

    Ix = np.dot(Img1,Gx)/2
    Iy = np.dot(Gy,Img1)/2




    # Time Derivative of image1 and image2
    It = Img1- Img2

    return Ix, Iy, It

def generateL (n):



    b1 = -2*np.ones((1,n))
    b1[0,0] =-1
    b1[0,n-1] =-1

    i = np.array(range(n)).flatten()
    print(b1.shape,i.shape)

    L0 = sp.sparse.coo_matrix((b1.flatten(), (i, i)), shape=(n, n))

    b2 = np.ones((1,n-1))
    i = np.array(range(n-1)).flatten()
    j = np.array(range(1,n)).flatten()
    L1 = sp.sparse.coo_matrix((b2.flatten(), (i, j)), shape=(n, n))

    b3 = np.ones((1, n - 1))
    L2 = sp.sparse.coo_matrix((b3.flatten(), (j, i)), shape=(n, n))

    L = L0 + L1 + L2

    return L


# Function to return the velocity vectors u and v using HornSchunck method with Laplace operator.
def HornSchunck(Img1, Img2,lam = 0.01):
    # lam is the regularization constant

    Ix, Iy, It = getGradients(Img1, Img2)


    J11 = Ix*Ix
    J22 = Iy*Iy
    J12 = Ix*Iy
    J21 = Iy*Ix
    J13 = Ix*It
    J23 = Iy*It


    n = len(Img1[0, :])  # width
    m = len(Img1[:, 0])  # height
    s = m * n  # total pixel number

    j11 = np.transpose(J11).flatten()
    j22 = np.transpose(J22).flatten()
    j12 = np.transpose(J12).flatten()
    j21 = np.transpose(J21).flatten()
    j13 = np.transpose(J13).flatten()
    j23 = np.transpose(J23).flatten()

    J11 = sp.sparse.spdiags(j11,0,s,s).tocsc()
    J22 = sp.sparse.spdiags(j22, 0, s, s).tocsc()
    J12 = sp.sparse.spdiags(j12, 0, s, s).tocsc()
    J21 = sp.sparse.spdiags(j21, 0, s, s).tocsc()

    T1 = generateL(n)
    T2 = generateL(m)


    coeff = (sp.sparse.kron(T1, sp.sparse.eye(m)) + sp.sparse.kron(sp.sparse.eye(n), T2)).tocsc()



    H1 = j13.reshape((s,1))
    H2 = j23.reshape((s,1))

    A11 = J11-lam*(coeff)
    A21 = J22-lam*(coeff)



    A1 = sp.sparse.hstack((J11-lam*(coeff),J12)).tocsc()
    A2 = sp.sparse.hstack((J21, J22-lam*(coeff))).tocsc()
    A = sp.sparse.vstack((A1,A2)).tocsc()
    H = -np.vstack((H1,H2))



    U = spsolve(A,H)

    #u, v = np.vsplit(U,2)

    u = np.reshape(U[0:s],(len(Img1[:, 0]), len(Img1[0, :])))
    v = np.reshape(U[s:2*s],(len(Img1[:, 0]), len(Img1[0, :])))

    return u,v




# Function to plot the optical FLow using HornSchunck
def PlotOF(Img1, Img2):
    Img1 = np.round(Img1/255,4)
    Img2 = np.round(Img2/255,4)

    u, v = HornSchunck(Img1, Img2,0.1)

    # plotting the velocity vectors in the image2
    plt.imshow(Img2, cmap='gray')


    # tracking each pixel
    for i in range (u.shape[0]):
        for j in range (u.shape[1]):
                ax = plt.axes()
                ax.arrow(j, i, u[i, j], v[i, j],color='r')
    figure()
    plt.show()



crop_size = (50, 50)

image1 = cv2.imread('basketball1.png',0)
image2 = cv2.imread('basketball2.png',0)
img1_new = cv2.resize(image1, crop_size, interpolation = cv2.INTER_CUBIC)
img2_new = cv2.resize(image2, crop_size, interpolation = cv2.INTER_CUBIC)
PlotOF(img1_new, img2_new)

