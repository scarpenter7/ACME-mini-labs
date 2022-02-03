# iterative_solvers.py
"""Volume 1: Iterative Solvers.
<Name> Sam Carpenter
<Class>
<Date> 4/6/21
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy import sparse

# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in range(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A

# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    # Initialize D inverse and x
    numIters = 0
    x = np.zeros_like(b)
    Dinv = np.zeros_like(A)
    diagEntries = np.array([1 / A[i, i] for i in range(len(A))])
    np.fill_diagonal(Dinv, diagEntries)
    absErrors = []

    # Iterate solve
    while numIters < maxiter:

        newx = x + Dinv @ (b - A@x)
        absError = la.norm(A@newx - b, ord=np.inf)
        absErrors.append(absError)
        numIters += 1
        if la.norm(x - newx, ord=np.inf) < tol:
            x = newx
            break
        x = newx

    # Plot results
    if plot:
        plt.semilogy(range(numIters), absErrors)
        plt.title("Convergence of Jacobi Method")
        plt.xlabel("Iteration")
        plt.ylabel("Abs Error of Approximation")
        plt.show()

    return x

# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    # Initialize D inverse and x
    numIters = 0
    x = np.zeros_like(b)
    absErrors = []

    # Iterate solve
    while numIters < maxiter:

        newx = x + np.array([(1/A[i, i])*(b[i] - A[i, :].T@x) for i in range(len(b))])
        absError = la.norm(A @ newx - b, ord=np.inf)
        absErrors.append(absError)
        numIters += 1
        if la.norm(x - newx, ord=np.inf) < tol:
            x = newx
            break
        x = newx

    # Plot results
    if plot:
        plt.semilogy(range(numIters), absErrors)
        plt.title("Convergence of Gauss Seidel Method")
        plt.xlabel("Iteration")
        plt.ylabel("Abs Error of Approximation")
        plt.show()
    return x

# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    # Initialize D inverse and x
    numIters = 0
    x = np.zeros_like(b)

    # Iterate solve
    while numIters < maxiter:
        newx = np.copy(x)

        for i in range(len(b)):
            rowstart = A.indptr[i]
            rowend = A.indptr[i + 1]
            # Multiply only the nonzero elements of the i-th row of A with the
            # corresponding elements of x.
            Aix = A.data[rowstart:rowend] @ newx[A.indices[rowstart:rowend]]
            newx[i] = x[i] + (b[i] - Aix)/A[i, i]

        numIters += 1
        if la.norm(x - newx, ord=np.inf) < tol:
            x = newx
            break
        x = newx
    return x


# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    # Initialize D inverse and x
    numIters = 0
    x = np.zeros_like(b)

    # Iterate solve
    while numIters < maxiter:
        newx = np.copy(x)

        for i in range(len(b)):
            rowstart = A.indptr[i]
            rowend = A.indptr[i + 1]
            # Multiply only the nonzero elements of the i-th row of A with the
            # corresponding elements of x.
            Aix = A.data[rowstart:rowend] @ newx[A.indices[rowstart:rowend]]
            # Add relaxation factor
            newx[i] = newx[i] + omega/A[i, i] *(b[i] - A.data[rowstart:rowend] @ newx[A.indices[rowstart:rowend]])

        numIters += 1
        if la.norm(newx - x, ord=np.inf) < tol:
            x = newx
            return x, True, numIters
        x = newx
    return x, False, numIters



# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """



# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """

# Tests:
def prob1Test():
    b = np.array([np.random.random() for _ in range(20)])
    A = diag_dom(len(b), len(b))
    x = jacobi(A, b)
    print(np.allclose(A@x, b))

def prob2Test():
    b = np.array([np.random.random() for _ in range(200)])
    A = diag_dom(len(b), len(b))
    x = jacobi(A, b, plot=True)
    print(np.allclose(A @ x, b))

def prob3Test():
    b = np.array([np.random.random() for _ in range(200)])
    A = diag_dom(len(b), len(b))
    x = gauss_seidel(A, b, plot=True)
    print(np.allclose(A @ x, b))

def prob4Test():
    A = sparse.csr_matrix(diag_dom(5000))
    b = np.random.random(5000)
    x = gauss_seidel_sparse(A, b, tol=1e-10)
    print(np.allclose(A @ x, b))
    print(A@x)
    print(b)
    print(la.norm(A@x - b, ord=1))

def prob5Test():
    A = sparse.csr_matrix(diag_dom(50000))
    b = np.random.random(50000)
    x, converged, iters = sor(A, b, 1.05, tol=1e-10)
    print("Converged: " + str(converged))
    print("Iterations: " + str(iters))
    print(np.allclose(A @ x, b))
    print(A@x)
    print(b)
    print(la.norm(A@x - b, ord=1))
    print(np.sum(np.abs(A@x - b)))


# prob1Test()
# prob2Test()
# prob3Test()
# prob4Test()
# prob5Test()

