# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Name> Sam Carpenter
<Class> Section 3
<Date> 11/2/20
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
import cmath
from scipy import linalg as la
from matplotlib import pyplot as plt


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q, R = la.qr(A, mode = "economic")
    return la.solve_triangular(R, Q.T @ b.T).T

#A = np.array([[1, 2, 3],[6, 5, 4],[2,6,3], [-5,5,8]])
#b = np.array([[3,6,2, -1]])
#print(least_squares(A, b))

#A = np.array([[0, 1],[1, 1],[2,1]])
#b = np.array([[6,0,0]])
#print(least_squares(A, b))



# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    data = np.load("housing.npy")
    m,n = data.shape
    ones = np.ones((m,1))
    years = data[:,0]
    A = np.column_stack((years, ones))
    b = data[:,1]
    solution = least_squares(A, b)

    plt.scatter(years, b)
    plt.plot(years * solution[0] + solution[1])
    plt.xlabel("Year (starting at 2000)")
    plt.ylabel("Price Index")
    plt.show()

#line_fit()


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    data = np.load("housing.npy")
    m, n = data.shape
    years = data[:, 0]
    x = np.linspace(0,years[-1], 100)
    priceIndexes = data[:,1]

    A3 = np.ones((m, 1))
    for i in range(1,4):
        newCol = data[:,0] ** i
        A3 = np.column_stack((newCol, A3))
    x3 = la.lstsq(A3, priceIndexes)[0]
    x3Poly = np.poly1d(x3)
    x3line = x3Poly(x)

    A6 = np.ones((m, 1))
    for i in range(1,7):
        newCol = data[:,0] ** i
        A6 = np.column_stack((newCol, A6))
    x6 = la.lstsq(A6, priceIndexes)[0]
    x6Poly = np.poly1d(x6)
    x6line = x6Poly(x)


    A9 = np.ones((m, 1))
    for i in range(1,9):
        newCol = data[:,0] ** i
        A9 = np.column_stack((newCol, A9))
    x9 = la.lstsq(A9, priceIndexes)[0]
    x9Poly = np.poly1d(x9)
    x9line = x9Poly(x)


    A12 = np.ones((m, 1))
    for i in range(1,13):
        newCol = data[:,0] ** i
        A12 = np.column_stack((newCol, A12))
    x12 = la.lstsq(A12, priceIndexes)[0]
    x12Poly = np.poly1d(x12)
    x12line = x12Poly(x)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(x, x3line)
    ax1.scatter(years, priceIndexes)
    ax1.set_title("Degree 3 Fit")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Price Index")

    ax2.plot(x, x6line)
    ax2.scatter(years, priceIndexes)
    ax2.set_title("Degree 6 Fit")
    ax2.set_xlabel("Year")
    ax3.set_ylabel("Price Index")

    ax3.plot(x, x9line)
    ax3.scatter(years, priceIndexes)
    ax3.set_title("Degree 9 Fit")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Price Index")

    ax4.plot(x, x12line)
    ax4.scatter(years, priceIndexes)
    ax4.set_title("Degree 12 Fit")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Price Index")

    plt.show()

#polynomial_fit()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    xk, yk = np.load("ellipse.npy").T
    A = np.column_stack((xk**2, xk, xk*yk, yk, yk**2))
    ones = np.ones_like(xk)

    a,b,c,d,e = la.lstsq(A, ones)[0]
    plot_ellipse(a,b,c,d,e)

    plt.plot(xk, yk, 'k*')
    plt.show()

#ellipse_fit()



# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    m,n = A.shape
    x0 = np.random.random(n)
    x0 /= la.norm(x0)
    xVectors = []
    xVectors.append(x0)
    for k in range(0, N):
        xk = np.dot(A, xVectors[-1])
        xk /= la.norm(xk)
        xVectors.append(xk)
        if la.norm(xk - xVectors[-2]) < tol:
            break
    return np.dot(xVectors[-1].T, np.dot(A, xVectors[-1])), xVectors[-1]

"""
A = np.array([[2, 3.5, 7, 9],[5, 1.2, 4.7, 0],[1, 0, 6.8, 0.3],[8.7, 1, 21, 3.4]])
eigs, vecs = la.eig(A)

leadingEig, eigVector = power_method(A)

print(leadingEig)
print(eigVector)

loc = np.argmax(eigs)
lamb, x = eigs[loc], vecs[:,loc]
print()
print(lamb)
print(x)

print(np.allclose(A @ x, lamb * x))
print(np.allclose(A @ eigVector, leadingEig * eigVector))
"""

# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    m,n = A.shape
    S = la.hessenberg(A)
    for k in range(N):
        Q,R = la.qr(S)
        S = R @ Q
    eigenvalues = []
    i = 0
    while i < n:
        if i == (n - 1) or abs(S[i + 1, i]) < tol:
            eigenvalues.append(S[i,i])
        else:
            b = -1 * S[i,i] + S[i+1,i+1]
            c = S[i,i]*S[i+1,i+1] - S[i,i+1] * S[i+1,i]
            x1 = (-b/(2)) + cmath.sqrt(b**2 - 4 * c) / 2
            x2 = (-b / (2)) - cmath.sqrt(b ** 2 - 4 * c) / 2
            eigenvalues.append(x1)
            eigenvalues.append(x2)
            i += 1
        i += 1
        eigenvalues.sort(reverse=True)
    return np.array(eigenvalues)
#B = np.array([[2, 3.2, 7],[3.2, 2, 4.7],[7, 4.7, 2]])
#print(B)

#print(qr_algorithm(B))
