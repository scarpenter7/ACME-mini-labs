"""Volume 2: Simplex

<Name> Sam Carpenter
<Date> 3/6/21
<Class>
"""

import numpy as np


# Problems 1-5
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        # Check for feasibility
        zeros = np.zeros_like(c)
        Ax = A @ zeros
        for i, p in enumerate(Ax):
            if p > b[i]:
                raise ValueError("This is infeasible at the origin.")

        # Store attributes in a dictionary
        self.D = self._generatedictionary(c, A, b)

    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        Abar = -np.hstack((A, np.eye(len(b))))
        cBar = np.hstack((c, np.zeros_like(b))).T
        bBar = np.hstack((np.zeros(1), b))

        # Combine Abar and cBar
        AcCombo = np.vstack((cBar, Abar))

        # combine the AcCombo with bBar
        D = np.hstack((bBar[:, np.newaxis], AcCombo))
        return D


    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        return np.argmax(self.D[0, 1:] < 0) + 1


    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        ratios = []
        for i in range(len(self.D[1:, 0])):
            if self.D[i + 1, index] >= 0:
                # Make sure we aren't dividing by zero
                ratios.append(np.inf)
            else:
                # append the ratio
                ratios.append(-self.D[i+1, 0]/self.D[i+1, index])

        return np.argmin(ratios) + 1

    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        pivCol = self._pivot_col()
        pivRow = self._pivot_row(pivCol)
        pivColEntries = self.D[1:, pivCol]

        # Check if all the entries in the pivot column are non-negative
        if all([p >= 0 for p in pivColEntries]):
            # If so, terminate algorithm, the solution is unbounded
            raise ValueError("The problem is unbounded.")

        divisor = -self.D[pivRow, pivCol]
        self.D[pivRow, :] = self.D[pivRow, :] / divisor

        # Zero out the column above and below the pivot row with row ops
        for i in range(len(self.D)):
            if i == pivRow:
                continue
            row = self.D[i, :]
            coeff = row[pivCol]
            self.D[i, :] = row + coeff * self.D[pivRow, :]

    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        # Pivot until all the values on top are positive
        while not all([d >= 0 for d in self.D[0, 1:]]):
            self.pivot()
        minValue = self.D[0, 0]

        # Get the indices of the dependent and independent variables
        depIndices = [i for i, p in enumerate(self.D[0, 1:]) if p == 0]
        indepIndices = [i for i, p in enumerate(self.D[0, 1:]) if p != 0]

        # Get the values of the dependent variables
        depVals = {}
        for depIndex in depIndices:
            col = self.D[1:, depIndex + 1]
            valIndex = [i for i, p in enumerate(col) if p == -1][0]
            depVal = round(self.D[1:, 0][valIndex], 1)
            depVals.update({depIndex: depVal})

        # Set all independent variables to 0
        indepVals = {indep: 0 for indep in indepIndices}

        return minValue, depVals, indepVals

# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    # Initialize data
    data = np.load(filename)
    resourceCoeffs = data['A']
    prices = -data['p']
    availableRes = data['m']
    demands = data['d']

    # Augment the constraints
    allConstraints = np.vstack((resourceCoeffs, np.eye(len(demands))))
    allResources = np.concatenate((availableRes, demands))

    # Solve using simplex
    solver = SimplexSolver(prices, allConstraints, allResources)
    solution = solver.solve()

    # Return the first 4 values of the dependent variables
    return np.array([solution[1][i] for i in range(4)])


# Tests

def prob1Test():
    b = np.array([1, 2, 3])
    c = np.array([-2, 1, -3])
    A = np.array([[2, -3, 1],
                  [-3, -1, 1],
                  [0, 2, -1]])

    solver = SimplexSolver(c, A, b)

def prob2Test():
    b = np.array([1, 2, 3])
    c = np.array([-2, 1, -3])
    A = np.array([[2, -3, 1],
                  [-3, -1, 1],
                  [0, 2, -1]])

    solver = SimplexSolver(c, A, b)

def prob3Test():
    b = np.array([2, 5, 7])
    c = np.array([-3, -2])
    A = np.array([[1, -1],
                  [3, 1],
                  [4, 3]])

    solver = SimplexSolver(c, A, b)
    print(solver.D)
    pivot = solver._pivot_col()
    print(pivot)
    row = solver._pivot_row(pivot)
    print(row)
    # Correct output: 1, 2

def prob4Test():
    b = np.array([2, 5, 7])
    c = np.array([-3, -2])
    A = np.array([[1, -1],
                  [3, 1],
                  [4, 3]])

    solver = SimplexSolver(c, A, b)
    solver.pivot()
    print(solver.D)

def prob5Test():
    b = np.array([2, 5, 7])
    c = np.array([-3, -2])
    A = np.array([[1, -1],
                  [3, 1],
                  [4, 3]])

    solver = SimplexSolver(c, A, b)
    print(solver.solve())
    print(solver.D)

def prob6Test():
    print(prob6())


# prob1Test()
# prob2Test()
# prob3Test()
# prob4Test()
# prob5Test()
# prob6Test()

