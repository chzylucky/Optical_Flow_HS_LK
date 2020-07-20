from scipy.signal import convolve2d
from pylab import *
import matplotlib.pyplot as plt
import cv2

#This is HS using iterations

# Function to return image and time derivatives fx, fy, ft
def getGradients(Img1, Img2):
    n = len(Img1[0, :])  # width: x direction
    m = len(Img1[:, 0])  # height: y direction
    s = m * n  # total pixel number

    Gx = zeros((m, m))
    Gy = zeros((n, n))

    for i in range(1, m - 1):
        Gy[i, i - 1] = -1
        print(i)
        Gy[i, i + 1] = 1

    for i in range(1, n - 1):
        Gx[i - 1, i] = -1
        Gx[i + 1, i] = 1

    fx = np.dot(Img1, Gx)
    fy = np.dot(Gy, Img1)

    #using Gaussian kernal and convolve to compute image derivative in x direction fx
    x = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
    #fx = (convolve2d(Img1, x) + convolve2d(Img2, x))/2
    fx = convolve2d(Img1, x,mode='same')
    print(fx.size)
    #fx = np.resize(fx, (len(Img1[:, 0]), len(Img1[0, :])))
    print(fx.size)

    #using Gaussian kernal and convolve to compute image derivative in y direction fy
    y = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]
    #fy = (convolve2d(Img1, y) + convolve2d(Img2, y))/2
    fy = convolve2d(Img1, y, mode='same')
    #fy = np.resize(fy, (len(Img1[:, 0]), len(Img1[0, :])))

    # Time Derivative of image1 and image2
    ft = Img2 - Img1

    return fx, fy, ft


# Function to return the velocity vectors u and v using HornSchunck method with Laplace operator.
def HornSchunck(Img1, Img2, iters= 8,lam = 0.001):
    # lam is the regularization constant

    fx, fy, ft = getGradients(Img1, Img2)

    u = zeros((len(Img1[:, 0]), len(Img1[0, :])))
    v = zeros((len(Img1[:, 0]), len(Img1[0, :])))

    #Using Laplace operator to find the second derivative
    L = [[0, 1/4, 0],[1/4, 0, 1/4],[0, 1/4, 0]]

    # Solving for the velocity vectors only for all Points; Iteration to reduce error
    for x in range(iters):

        uAvg = convolve2d(u,L)
        vAvg = convolve2d(v, L)
        uAvg.resize(len(Img1[:, 0]), len(Img1[0, :]))
        vAvg.resize(len(Img1[:, 0]), len(Img1[0, :]))

        # common part of update step
        pd = (fx * uAvg + fy * vAvg + ft) / (lam ** 2 + fx ** 2 + fy ** 2)

        #iterative part
        u = uAvg - fx * pd
        v = vAvg - fy * pd

    return u, v


# Function to plot the optical FLow using HornSchunck
def PlotOF(Img1, Img2):
    # Using openCV inbuilt function goodFeaturestoTrack to find the corners in the image1
    corners = cv2.goodFeaturesToTrack(Img1, 200, 0.03, 7)

    # detected corners converted to type int
    corners = np.int0(corners)
    print(corners.shape)


    u, v = HornSchunck(Img1, Img2,14,250)


    # plotting the velocity vectors in the image2
    plt.imshow(Img2, cmap='gray')

    # just tracking the good features/corners
    for i in range (u.shape[0]):
        for j in range (u.shape[1]):

                ax = plt.axes()
                #n += 1
                ax.arrow(j, i, u[i, j], v[i, j], head_width=0.3, head_length=0.3, color='r')
    figure()
    plt.show()




crop_size = (50, 50)
image1 = cv2.imread('basketball1.png',0)
image2 = cv2.imread('basketball2.png',0)

img1_new = cv2.resize(image1, crop_size, interpolation = cv2.INTER_CUBIC)
img2_new = cv2.resize(image2, crop_size, interpolation = cv2.INTER_CUBIC)
PlotOF(img1_new, img2_new)

