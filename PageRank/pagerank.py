# solutions.py
"""Volume 1: The Page Rank Algorithm.
<Name> Sam Carpenter
<Class>
<Date> 3/16/21
"""

import numpy as np
import networkx as nx
from scipy import linalg as la

# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        # fix sinks
        m, n = A.shape
        for col in range(n):
            currCol = A[:, col]
            if np.allclose(currCol, np.zeros_like(currCol)):
                A[:, col] = np.ones_like(currCol)

        # Scale the columns, save A as attribute
        A = A / np.sum(A, axis=0)
        self.A = A
        self.n = n

        # Check the labels and save them as attribute
        if labels is None:
            labels = list(range(n))
        elif len(labels) != n:
            raise ValueError("Number of labels is not equal to the number of nodes in the graph.")
        self.labels = labels

    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # See textbook equation
        G = np.eye(self.n) - epsilon * self.A
        b = np.ones(self.n) * (1 - epsilon) / self.n
        p = la.solve(G, b)
        return {self.labels[i]: p[i] for i in range(len(p))}

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # See equations in the textbook
        B = epsilon * self.A + np.ones_like(self.A) * (1 - epsilon) / self.n
        eVals, eVects = np.linalg.eig(B)
        p = np.abs(eVects[:, 0] / la.norm(eVects[:, 0], ord=1))
        return {self.labels[i]: p[i] for i in range(len(p))}

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        p = np.ones(self.n) / self.n
        numIters = 0

        # Iterate until we exceed maxiter or the difference is smaller than a certain tolerance.
        while numIters < maxiter:
            nextP = epsilon * self.A @ p + np.ones(self.n) * (1 - epsilon) / self.n
            numIters += 1
            if la.norm(p - nextP, ord=1) < tol:
                break
            p = nextP
        return {self.labels[i]: p[i] for i in range(len(p))}


# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    """labels = d.keys()
    vals = d.values()
    vals = [-v for v in vals]
    data = list(zip(vals, labels))
    # Sort by rank from greatest to lowest.
    data.sort()"""
    labels = [(-val, key) for key, val in zip(d.keys(), d.values())]
    labels.sort()
    return [l[1] for l in labels]

# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    file = open(filename)
    txt = file.read()
    lines = txt.split('\n')

    # Get the labels first
    labels = set()
    for i, line in enumerate(lines):
        IDs = line.split('/')
        IDs = [id for id in IDs]
        for ID in IDs:
            labels.add(ID)

    # Initialize the adjacency matrix
    labelsList = list(labels)
    labelsList.sort()
    labels = {ID: i for i, ID in enumerate(labelsList)}
    n = len(labels)
    A = np.zeros((n, n))

    # Build Matrix
    for line in lines:
        IDs = line.split('/')
        IDs = [id for id in IDs]
        startID = IDs[0]
        startIndex = labels.get(startID)
        indices = getIndices(labels, IDs[1:])
        for index in indices:
            A[index, startIndex] += 1

    graph = DiGraph(A, labelsList)
    ranks = get_ranks(graph.itersolve(epsilon=epsilon))
    return ranks

def getIndices(labels, IDs):
    indices = []
    for ID in IDs:
        #if ID in labels.keys():
        index = labels.get(ID)
        indices.append(index)
    return indices



# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    file = open(filename)
    lines = file.read().split('\n')
    teams = set()

    # Get team names first
    for i, line in enumerate(lines):
        if line == "":
            break
        if i == 0:
            continue
        colleges = line.split(',')
        winner = colleges[0]
        loser = colleges[1]
        teams.add(winner)
        teams.add(loser)

    teams = list(teams)
    n = len(teams)

    A = np.zeros((n, n))

    # Build Matrix
    for i, line in enumerate(lines):
        if line == "":
            break
        if i == 0:
            continue
        colleges = line.split(',')
        winner = colleges[0]
        loser = colleges[1]
        colIndex = teams.index(loser)
        rowIndex = teams.index(winner)
        A[rowIndex, colIndex] += 1

    # Compute ranks
    graph = DiGraph(A, teams)
    ranks = get_ranks(graph.itersolve(epsilon=epsilon))
    return ranks

# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    file = open(filename, encoding="utf-8")
    movies = file.read().split('\n')
    DG = nx.DiGraph()

    # Build the graph
    for movie in movies:
        info = movie.split('/')
        actors = info[1:]
        # Loop through combinations
        for i, actor in enumerate(actors):
            for a2 in actors[i+1:]:
                if not DG.has_edge(a2, actor):
                    DG.add_edge(a2, actor, weight=1)
                else:
                    DG[a2][actor]['weight'] += 1

    # Compute page ranks
    d = nx.pagerank(DG, alpha=epsilon)
    ranks = get_ranks(d)
    return ranks


# Tests

def prob1Test():
    A = np.array(([[0, 0, 0, 0],
                   [1, 0, 1, 0],
                   [1, 0, 0, 1],
                   [1, 0, 1, 0]]))

    graph = DiGraph(A)
    print(graph.A)

def prob2Test():
    A = np.array(([[0, 0, 0, 0],
                   [1, 0, 1, 0],
                   [1, 0, 0, 1],
                   [1, 0, 1, 0]]))

    graph = DiGraph(A)
    print(graph.linsolve())
    print(graph.eigensolve())
    print(graph.itersolve())

def prob3Test():
    A = np.array(([[0, 0, 0, 0],
                   [1, 0, 1, 0],
                   [1, 0, 0, 1],
                   [1, 0, 1, 0]]))

    graph = DiGraph(A, labels=['a', 'b', 'c', 'd'])
    d = graph.linsolve()
    print(get_ranks(d))

def prob4Test():
    print("My answer:")
    myAns = rank_websites(epsilon=0.5)[:20]
    print(myAns)
    print("Correct answer:")
    ans = ['98595','32791','178606','28392','77323','92715','26083','130094','99464','12846','106064','332','31328','86049','123900','74923','119538','90571','116900','139197']
    print(ans)
    print(ans == myAns)
    print()

def prob4Test2():
    print("My answer:")
    myAns = rank_websites(epsilon=0.97)[:20]
    print(myAns)
    print("Correct answer:")
    ans = ['98595', '32791', '28392', '77323', '92715', '26083', '130094', '99464', '12846', '106064', '332', '31328', '86049', '123900', '74923', '90571', '119538', '116900', '139197', '114623']
    print(ans)
    print(ans == myAns)
    print()

def prob4Test3():
    print("My answer:")
    myAns = rank_websites(epsilon=0.35)[:20]
    print(myAns)
    print("Correct answer:")
    ans = ['13787', '15672', '4815', '6886', '3881', '7314', '159964', '12846', '40349', '104202', '7002', '32791', '68080', '240318', '29960', '67827', '94526', '130094', '332', '7027']
    print(ans)
    print(ans == myAns)
    print()

def prob5Test():
    print(rank_ncaa_teams("ncaa2010.csv"))

def prob6Test():
    print("My answer:")
    print(rank_actors(epsilon=0.55)[:20])
    print("Correct answer:")
    print(['Leonardo DiCaprio', 'Jamie Foxx', 'Robert De Niro', 'Christoph Waltz', 'Al Pacino', 'Tom Hanks', 'Christian Bale', 'Ben Kingsley', 'Brad Pitt', 'Ralph Fiennes', 'Liam Neeson', 'Antonella Attili', 'Matt Damon', 'Diahnne Abbott', 'Tom Hardy', 'Morgan Freeman', 'Gary Oldman', 'Ryan Gosling', 'Harrison Ford', 'Karen Gillan'])

def prob6Test2():
    print("My answer:")
    print(rank_actors(epsilon=0.71)[:20])
    print("Correct answer:")
    """print(['Leonardo DiCaprio', 'Jamie Foxx', 'Robert De Niro', 'Christoph Waltz', 'Al Pacino', 'Tom Hanks',
           'Christian Bale', 'Ben Kingsley', 'Brad Pitt', 'Ralph Fiennes', 'Liam Neeson', 'Antonella Attili',
           'Matt Damon', 'Diahnne Abbott', 'Tom Hardy', 'Morgan Freeman', 'Gary Oldman', 'Ryan Gosling',
           'Harrison Ford', 'Karen Gillan'])"""


# prob1Test()
# prob2Test()
# prob3Test()
# prob4Test()
# prob4Test2()
# prob4Test3()
# prob5Test()
# prob6Test()
# prob6Test2()