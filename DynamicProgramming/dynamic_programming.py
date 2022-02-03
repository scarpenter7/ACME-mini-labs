# dynamic_programming.py
"""Volume 2: Dynamic Programming.
<Name> Sam Carpenter
<Class>
<Date> 4/1/21
"""

import numpy as np
from matplotlib import pyplot as plt


def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    vals = [0]
    val = 0
    # Iterate over a list of reversed indices
    for n in reversed(range(1, N + 1)):
        if n == N: # skip for the first one
            continue
        val = max(val, (n/(n+1))*val + 1/N)
        vals.append(val)
    # Get the max val and return N - n
    n = np.argmax(vals)
    return vals[n], N - n

# Problem 2
def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    Ns = range(3, M + 1)
    percentages = []
    bestVals = []
    # Perform prob1 for every value of N
    for N in Ns:
        bestVal, bestT = calc_stopping(N)
        stopPercentage = bestT/N
        percentages.append(stopPercentage)
        bestVals.append(bestVal)

    MStopPercentage = percentages[-1]

    # Plot Results
    plt.plot(Ns, percentages, label="Stop percentages")
    plt.plot(Ns, bestVals, label="Max probability")
    plt.legend()
    plt.xlabel("N")
    plt.show()

    return MStopPercentage


# Problem 3
def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    # Initialize the w vector and C matrix
    w = np.array([i/N for i in range(N + 1)])
    C = np.zeros((N + 1, N + 1))
    rows, cols = C.shape

    # Build the rows and columns of C in lower triangular form
    for row in range(rows):
        wIndex = row
        for col in range(row):
            C[row, col] = u(w[wIndex])
            wIndex -= 1
    return C


# Problems 4-6
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    # Problem 4
    A = np.zeros((N + 1, T + 1))
    A[:, -1] = u(np.linspace(0, 1, N + 1))
    P = np.zeros((N + 1, T + 1))

    # Problem 5
    C = get_consumption(N, u)
    for t in range(T):
        # Current Value matrix
        CV = np.array([[C[i, j] + B * A[j, T - t] if j <= i else 0 for j in range(N + 1)] for i in range(N + 1)])

        A[:, T - t - 1] = np.max(CV, axis=1)
        # Problem 6
        P[:, -1] = np.linspace(0, 1, N + 1)
        for j in range(N + 1):
            P[j, T - t - 1] = (j - np.argmax(CV[j, :])) / N

    return A, P



# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    # use previous function
    A, P = eat_cake(T, N, B, u)
    policy = []

    # Start from bottom left and go up 1 and right 1 to get policy
    rows, cols = P.shape
    for j, i in enumerate(reversed(range(1, rows))):
        policy.append(P[i, j])
    return np.array(policy)



# Tests
def prob1Test():
    print(calc_stopping(25))

def prob2Test():
    print(graph_stopping_times(200))

def prob3Test():
    print(get_consumption(4))

def prob456Test():
    T = 3
    N = 4
    B = .9
    print(eat_cake(T, N, B))

def prob7Test():
    T = 3
    N = 4
    B = .9
    print(find_policy(T, N, B))

# prob1Test()
# prob2Test()
# prob3Test()
# prob456Test()
# prob7Test()