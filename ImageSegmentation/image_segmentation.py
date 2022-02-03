# image_segmentation.py
"""Volume 1: Image Segmentation.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from imageio import imread
from scipy import sparse
from scipy.sparse import linalg


# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    L = -1 * A
    m,n = A.shape
    colSums = A.sum(axis=0)
    for i in range(n):
        L[i,i] = colSums[i]
    return L
#A = np.array([[0, 3.5, 7, 9],[5, 0, 4.7, 0],[1, 0, 0, 0.3],[8.7, 1, 21, 0]])
#print(A)
#print(laplacian(A))
#print(sparse.csgraph.laplacian(A))

# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    L = laplacian(A)
    #print(L)
    eigs, eVectors = la.eig(L)
    eigs = np.real(eigs)
    numConnectedComponents = 0

    #count connected components

    for i in range(len(eigs)):
        if eigs[i] < tol:
            eigs[i] = 0
        if eigs[i] == 0:
            numConnectedComponents += 1

    #Find the second smallest eigenvalue (nonzero)

    eigs = [value for value in eigs if value != 0]
    return numConnectedComponents, min(eigs)


#A3 = np.array([[0, 1, 0, 0, 1, 1],
              #[1, 0, 1, 0, 1, 0],
              #[0, 1, 0, 1, 0, 0],
              #[0, 0, 1, 0, 1, 1],
              #[1, 1, 0, 1, 0, 0],
              #[1, 0, 0, 1, 0, 0]])
#print(connectivity(A))

#A2 = np.array([[0, 3, 0, 0, 0, 0],
              #[3, 0, 0, 0, 0, 0],
              #[0, 0, 0, 1, 0, 0],
              #[0, 0, 1, 0, 2, .5],
              #[0, 0, 0, 2, 0, 1],
              #[0, 0, 0, .5, 1, 0]])
#print(connectivity(A2))



# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]




# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        self.image = imread(filename)
        scaled = self.image / 255
        self.scaled = scaled
        #print(self.image.shape)
        self.grayScale = True
        self.brightnessFlat = None
        #Check if it's a color image
        if self.image.ndim == 3: #then it's a color image
            self.grayScale = False
            brightness = scaled.mean(axis=2)
            self.brightnessFlat = np.ravel(brightness)
        else:
            self.brightnessFlat = np.ravel(scaled)

    # Problem 3
    def show_original(self):
        """Display the original image."""
        if self.grayScale:
            plt.imshow(self.image, cmap="gray")
            plt.show()
        else:
            plt.imshow(self.image)
            plt.show()


    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        A = sparse.lil_matrix((len(self.brightnessFlat), len(self.brightnessFlat)))
        D = []
        m, n = self.image.shape[:2]
        for i in range(len(self.brightnessFlat)):
            neighbors, distances = get_neighbors(i, r, m, n)
            brightnessDiffs = [abs(self.brightnessFlat[i] - self.brightnessFlat[b2]) for b2 in neighbors]
            weights = self.computeWeights(brightnessDiffs, distances, sigma_B2, sigma_X2, r)
            A[i, neighbors] = weights
            D.append(sum(weights))
        return A.tocsc(), np.array(D)

    def computeWeights(self, brightnessDiffs, distances, sigma_B, sigma_X, r):
        weights = []
        for i, distance in enumerate(distances):
            if distance < r:
                weight = np.exp(-(brightnessDiffs[i] / sigma_B) - distance / sigma_X)
                weights.append(weight)
            else:
                weights.append(0)
        return weights

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        L = sparse.csgraph.laplacian(A)
        D12 = sparse.diags(np.power(D, -.5))
        M = D12 @ L @ D12

        #Compute eigenvector then reshape

        eVector = sparse.linalg.eigsh(M, which="SM", k=2)[1][:,1]
        m, n = (self.image.shape[0], self.image.shape[1])
        eVectorMatrix = eVector.reshape((m, n))
        mask = eVectorMatrix > 0
        return mask

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        #Image 1
        ax1 = plt.subplot(131)
        if self.grayScale:
            ax1.imshow(self.image, cmap="gray")
        else:
            ax1.imshow(self.image)

        A, D = self.adjacency(r, sigma_B, sigma_X)
        mask = self.cut(A, D)
        # Image 2
        ax2 = plt.subplot(132)
        if self.grayScale:
            ax2.imshow(self.image * mask, cmap="gray")
        else:
            mask3 = np.dstack([mask, mask, mask])
            ax2.imshow(self.image * mask3)
        #Image 3
        ax3 = plt.subplot(133)
        if self.grayScale:
            newImageAlt = np.multiply(self.image, ~mask)
            ax3.imshow(newImageAlt, cmap="gray")
        else:
            mask3 = np.dstack((~mask, ~mask, ~mask))
            newImageAlt = np.multiply(self.image, mask3)
            ax3.imshow(newImageAlt)


        plt.show()


#if __name__ == '__main__':
#     ImageSegmenter("dream_gray.png").segment()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()
#imgSeg = ImageSegmenter("john_frusciante.png")
#imgSeg.show_original()

    #imgSeg = ImageSegmenter("blue_heart.png")
    #imgSeg.show_original()
    #solutionA = sparse.load_npz("HeartMatrixA.npz")
    #solutionD = np.load("HeartMatrixD.npy")


    #A, D = imgSeg.adjacency()
    #print(A.shape)
    #print(solutionA.shape)
    #print(D.shape)
    #print(solutionD.shape)
    #print(np.allclose(A.data, solutionA.data))
    #print(np.allclose(D.data, solutionD.data))

    #imgSeg2 = ImageSegmenter("dream_gray.png")
    #imgSeg2.segment()
