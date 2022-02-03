# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Name> Sam Carpenter
<Class>
<Date> 3/11/21
"""

import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # First we'll initialize the objective
    # We can declare x with its size and sign
    x = cp.Variable(3, nonneg=True)
    c = np.array([2, 1, 3])
    objective = cp.Minimize(c.T @ x)

    # Then we'll write the constraints
    A1 = np.array([1, 2, 0])
    A2 = np.array([0, 1, -4])
    A3 = np.array([2, 10, 3])
    P = np.eye(3)
    constraints = [A1 @ x <= 3, A2 @ x <= 1, A3 @ x >= 12, P @ x >= 0]  # This must be a list

    # Assemble the problem and then solve it
    problem = cp.Problem(objective, constraints)
    optimalVal = problem.solve()
    return x.value, optimalVal

# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    m, n = A.shape
    # First we'll initialize the objective
    # We can declare x with its size and sign
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(x, 1))

    # Then we'll write the constraints
    constraints = [A@x == b]  # This must be a list

    # Assemble the problem and then solve it
    problem = cp.Problem(objective, constraints)
    optimalVal = problem.solve()
    return x.value, optimalVal

# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # First we'll initialize the objective
    # We can declare x with its size and sign
    x = cp.Variable(6, nonneg=True)
    c = np.array([4, 7, 6, 8, 8, 9])
    objective = cp.Minimize(c.T @ x)

    # Then we'll write the constraints
    A1 = np.array([1, 1, 0, 0, 0, 0])
    A2 = np.array([0, 0, 1, 1, 0, 0])
    A3 = np.array([0, 0, 0, 0, 1, 1])
    B1 = np.array([1, 0, 1, 0, 1, 0])
    B2 = np.array([0, 1, 0, 1, 0, 1])
    P = np.eye(6)
    constraints = [A1 @ x <= 7, A2 @ x <= 2, A3 @ x <= 4, B1 @ x == 5, B2 @ x == 8, P @ x >= 0]  # This must be a list

    # Assemble the problem and then solve it
    problem = cp.Problem(objective, constraints)
    optimalVal = problem.solve()
    return x.value, optimalVal

# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # g = lambda x: (3/2)*x[0]**2 + 2*x[0]*x[1] + x[0]*x[2] + 2*x[1]**2 + 2*x[1]*x[2] + (3/2)*x[2]**2 + 3*x[0] + x[2]

    # Initialize Q and r
    Q = np.array([[3, 2, 1], [2, 4, 2], [1, 2, 3]])
    r = np.array([3, 0, 1])
    x = cp.Variable(3)

    # Solve the quadratic problem
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, Q) + r.T @ x))
    optimalVal = prob.solve()
    return x.value, optimalVal

# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    m, n = A.shape
    # First we'll initialize the objective
    # We can declare x with its size and sign
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(A @ x - b, 2))

    # Then we'll write the constraints
    constraints = [cp.sum(x) == 1, x >= 0]

    # Assemble the problem and then solve it
    problem = cp.Problem(objective, constraints)
    optimalVal = problem.solve()
    return x.value, optimalVal

# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # First we'll initialize the objective
    data = np.load("food.npy", allow_pickle=True)
    n = len(data)
    x = cp.Variable(n, nonneg=True)
    # Get prices in the first column
    c = data[:, 0]
    servs = data[:, 1]
    objective = cp.Minimize(c.T @ x)

    # Multiply each feature by the amount of servings there are
    cals = data[:, 2] * servs
    fat = data[:, 3] * servs
    sugar = data[:, 4] * servs
    calcium = data[:, 5] * servs
    fiber = data[:, 6] * servs
    protein = data[:, 7] *servs
    P = np.eye(n)
    constraints = [cals @ x <= 2000, fat @ x <= 65, sugar @ x <= 50, calcium @ x >= 1000, fiber @ x >= 25, protein @ x >= 46, P @ x >= 0]  # This must be a list

    # Assemble the problem and then solve it
    problem = cp.Problem(objective, constraints)
    optimalVal = problem.solve()
    return np.array(x.value), optimalVal


# Tests
def prob1Test():
    print(prob1())

def prob2Test():
    A = np.array([[1, 2, 1, 1],
                  [0, 3, -2, -1]])

    b = np.array([7, 4])

    print(l1Min(A, b))

def prob3Test():
    print(prob3())

def prob4Test():
    print(prob4())

def prob5Test():
    A = np.array([[1, 2, 1, 1],
                  [0, 3, -2, -1]])

    b = np.array([7, 4])
    print(prob5(A, b))

def prob6Test():
    print(prob6())

# prob1Test()
# prob2Test()
# prob3Test()
# prob4Test()
# prob5Test()
# prob6Test()