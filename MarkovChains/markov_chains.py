# markov_chains.py
"""Volume 2: Markov Chains.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        (fill this out)
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            #>>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        if not np.allclose(A.sum(axis=0), np.ones(A.shape[1])):
            raise ValueError("A is not column stochastic")
        m,n = A.shape
        if states is None:
            states = range(n)
        self.stateMap = {}
        self.states = states
        for i, state in enumerate(states):
            self.stateMap.update({state:i})
        self.A = A

    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        colIndex = self.stateMap.get(state)
        outcome = np.random.multinomial(1, self.A[:,colIndex])
        outcomeState = None
        m,n = self.A.shape
        for i in range(n):
            if outcome[i] == 1:
                outcomeState = self.states[i]
        return outcomeState

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        pathTravelled = [start]
        currentState = start
        for _ in range(N - 1):
            currentState = self.transition(currentState)
            pathTravelled.append(currentState)
        return pathTravelled

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        pathTravelled = [start]
        currentState = start
        while currentState != stop:
            currentState = self.transition(currentState)
            pathTravelled.append(currentState)
        return pathTravelled

    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        currentStateDistribution = self.A[:,0]
        for i in range(maxiter):
            prevStateDistribution = currentStateDistribution
            currentStateDistribution = self.A @ currentStateDistribution
            if la.norm(currentStateDistribution - prevStateDistribution) < tol:
                return currentStateDistribution
        raise ValueError("A^k does not converge.")

#A = np.array(([0.5, 0.3, 0.1, 0], [0.3, 0.3, 0.3, 0.3], [0.2, 0.3, 0.4, 0.5], [0, 0.1, 0.2, 0.2]))
#states = ["hot", "mild", "cold", "freezing"]
#mChain = MarkovChain(A, states)
#print(mChain.walk("mild", 10))
#print(mChain.path("hot", "freezing"))
#print(mChain.steady_state())

#Asmall = np.array([[.7, .6],[.3, .4]])
#statesSmall = ["hot", "cold"]
#mChainSmall = MarkovChain(Asmall, statesSmall)
#stats = mChainSmall.walk("cold", 1000)
#print(sum([1 for stat in stats if stat == "hot"]))

#print(mChainSmall.steady_state())

class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        #Initialize the file contents and labels
        file = open(filename)
        self.contents = file.read()
        self.lines = self.contents.split('\n')
        self.words = set()
        for line in self.lines:
            words = line.split(' ')
            for word in words:
                self.words.add(word)
        self.states = list(self.words)
        self.states.insert(0, "$tart")
        self.states.append("$top")
        self.stateMap = {}
        for i, state in enumerate(self.states):
            self.stateMap.update({state: i})

        #Construct Transition Matrix
        self.A = np.zeros((len(self.states), len(self.states)))
        for line in self.lines:
            words = line.split(' ')
            words.insert(0, "$tart")
            words.append("$top")
            for i in range(1, len(words)):
                word1 = words[i - 1]
                word2 = words[i]
                index1 = self.stateMap.get(word1)
                index2 = self.stateMap.get(word2)
                self.A[index2, index1] += 1


        #Normalize the column sums
        self.A[-1, -1] = 1
        for i in range(len(self.states)):
            self.A[:, i] /= sum(self.A[:, i])

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            #>>> yoda = SentenceGenerator("yoda.txt")
            #>>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        path = self.path("$tart", "$top")
        path.pop(0)
        path.pop(-1)
        message = " ".join(path)
        return message

#generator = SentenceGenerator("yoda.txt")
#for _ in range(5):
#   print(generator.babble())