
from pylab import *
import matplotlib.pyplot as plt
import cv2


# Function to return image and time derivatives fx, fy, ft
def getGradients(Img1, Img2):

    # #using Gaussian kernal and convolve to compute image derivative in x direction fx
    # x = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
    # #fx = (convolve2d(Img1, x) + convolve2d(Img2, x))/2
    # fx = convolve2d(Img1, x,mode='same')
    # print(fx.size)
    # #fx = np.resize(fx, (len(Img1[:, 0]), len(Img1[0, :])))
    # print(fx.size)
    #
    # #using Gaussian kernal and convolve to compute image derivative in y direction fy
    # y = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]
    # #fy = (convolve2d(Img1, y) + convolve2d(Img2, y))/2
    # fy = convolve2d(Img1, y, mode='same')
    # #fy = np.resize(fy, (len(Img1[:, 0]), len(Img1[0, :])))

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

    # Time Derivative of image1 and image2
    ft = Img1 - Img2

    return fx, fy, ft


# Function to return the velocity vectors u and v by solving the matrices using Least Square Fit Method
def LucasKannade(Img1, Img2, corners):
    fx, fy, ft = getGradients(Img1, Img2)

    u = zeros((len(Img1[:, 0]), len(Img1[0, :])))
    v = zeros((len(Img1[:, 0]), len(Img1[0, :])))

    # Solving for the velocity vectors only for the good Feature Points i.e corners detected using openCV function
    for x in corners:
        j, i = x.ravel()

        # this creates a 3x3 matrix for each pixel in the good feature point
        Fx = fx[(i - 1):(i + 2), (j - 1):(j + 2)]
        Fy = fy[(i - 1):(i + 2), (j - 1):(j + 2)]
        Ft = ft[(i - 1):(i + 2), (j - 1):(j + 2)]

        # transposed as the ravel function would straighten up the 3x3 image to 9x1 reading columnwise
        Fx = np.transpose(Fx)
        Fy = np.transpose(Fy)
        Ft = np.transpose(Ft)

        Fx = Fx.flatten()
        Fy = Fy.flatten()
        Ft = -Ft.flatten()

        # 9x2 matrix
        A = np.transpose([Fx, Fy])

        # Least square fit to solve for unknown
        product1 = np.dot(np.transpose(A), A)
        product2 = np.dot(linalg.pinv(product1), np.transpose(A))
        U = np.dot(product2, Ft)

        # 2x1 matrix containing the velocity vector for a single pixel point
        u[i, j] = U[0]
        v[i, j] = U[1]

    return u, v


# Function to plot the optical FLow using Lucas-Kannade
def PlotOF(Img1, Img2):
    # Using openCV inbuilt function goodFeaturestoTrack to find the corners in the image2
    corners = cv2.goodFeaturesToTrack(Img1, 200, 0.03, 7)

    # detected corners converted to type int
    corners = np.int0(corners)
    print(corners.shape)


    u, v = LucasKannade(Img1, Img2, corners)
    #n = 0

    # plotting the velocity vectors in the image2
    plt.imshow(Img2, cmap='gray')
    for i in range (u.shape[0]):
        for j in range (u.shape[1]):

                ax = plt.axes()
                #n += 1
                ax.arrow(j, i, u[i, j], v[i, j], head_width=0.3, head_length=0.35, color='b')
    figure()
    plt.show()




crop_size = (50, 50)
image1 = cv2.imread('basketball1.png',0)
image2 = cv2.imread('basketball2.png',0)

img1_new = cv2.resize(image1, crop_size, interpolation = cv2.INTER_CUBIC)
img2_new = cv2.resize(image2, crop_size, interpolation = cv2.INTER_CUBIC)
PlotOF(img1_new, img2_new)
