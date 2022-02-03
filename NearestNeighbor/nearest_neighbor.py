# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
<Name> Sam Carpenter
<Class> Section 3
<Date> 10/22/20
"""

import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree
import scipy.stats as stats
from matplotlib import pyplot as plt

# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    rowNorms = la.norm(X - z, axis=1)
    minIndex = list(rowNorms).index(min(rowNorms))
    minx = X[minIndex]
    return minx, min(rowNorms)

A = np.array([[1, 2, 3],[6, 5, 4],[2,6,3]])
z = np.array([[3,6,2]])
#print(exhaustive_search(A,z))

# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):
        if type(x) != type(np.array([])):
            raise TypeError("input was not a NumPy array")
        self.value = x
        self.pivot = None
        self.right = None
        self.left = None

# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        newNode = KDTNode(data)
        #If the tree is empty
        if self.root is None:
            self.root = newNode
            self.k = len(data)
            newNode.pivot = 0
            return
        #If the length of data doesn't match the k value, raise value error
        if len(data) != self.k:
            raise ValueError("Length of data does not match the k value of the KDTree.")

        #If the tree is not empty, find the parent and then insert in the correct position
        parentNode = self.findParent(list(data))
        pivotValue = parentNode.pivot

        if data[pivotValue] < parentNode.value[pivotValue]:
            parentNode.left = newNode
        else:
            parentNode.right = newNode
        #after inserting, set the new node's pivot value
        newNode.pivot = (pivotValue + 1) % self.k


    def findParent(self, data):
        def _step(current, prevNode):
            if current is None:                     # Base case 1: dead end.
                return prevNode
            elif np.allclose(data, current.value):
                raise ValueError("value already contained in the tree") # Base case 2: identical node found!
            elif data[current.pivot] < list(list(current.value))[current.pivot]:
                return _step(current.left, current)          # Recursively search left.
            else:
                return _step(current.right, current)         # Recursively search right.
        return _step(self.root, None)

    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        def _KDSearch(current, nearest, dMin):
            if current is None:
                return  nearest, dMin
            x = current.value
            i = current.pivot
            if la.norm(x - z) < dMin:
                nearest = current
                dMin = la.norm(x - z)
            if z[i] < x[i]:
                nearest, dMin = _KDSearch(current.left, nearest, dMin)
                if z[i] + dMin >= x[i]:
                    nearest, dMin = _KDSearch(current.right, nearest, dMin)
            else:
                nearest, dMin = _KDSearch(current.right, nearest, dMin)
                if z[i] - dMin <= x[i]:
                    nearest, dMin = _KDSearch(current.left, nearest, dMin)
            return nearest, dMin
        node, dMin = _KDSearch(self.root, self.root, la.norm(self.root.value - z))
        return node.value, dMin

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)
"""
kdt = KDT()
print(KDTree)
node1 = np.array([3,1,4])
print(node1)
kdt.insert(node1)
print(kdt)
print(kdt.k)
kdt.insert(np.array([1,2,7]))
kdt.insert(np.array([4,3,5]))
kdt.insert(np.array([2,0,3]))
kdt.insert(np.array([2,4,5]))
kdt.insert(np.array([6,1,4]))
kdt.insert(np.array([1,4,3]))
kdt.insert(np.array([0,5,7]))
kdt.insert(np.array([5,2,5]))
print(kdt)
"""
"""
print(exhaustive_search(A,z))
kdt = KDT()
kdt.insert(np.array([[1,2,3]]))
kdt.insert(np.array([[6,5,4]]))
kdt.insert(np.array([[2,6,3]]))
#print(la.norm(kdt.root.value, z))
#print(kdt.query(z))
"""

# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.kdt = None
        self.labels = None

    def fit(self, trainingSet, yLabels):
        kdt = KDTree(trainingSet)
        self.kdt = kdt
        self.labels = yLabels

    def predict(self, z):
        queryResult, indices = self.kdt.query(z, self.n_neighbors)
        selectedNeighbors = self.labels[indices]
        return stats.mode(selectedNeighbors)[0][0]


# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    data = np.load(filename)
    X_train = data["X_train"].astype(np.float)  # Training data
    y_train = data["y_train"]  # Training labels
    X_test = data["X_test"].astype(np.float)  # Test data
    y_test = data["y_test"]
    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(X_train, y_train)
    numCorrect = 0
    for i,test in enumerate(X_test,0):
        label = classifier.predict(test)
        if label == y_test[i]:
            numCorrect += 1
    percentage = numCorrect / len(X_test)

    plt.imshow(X_test[0].reshape((28, 28)), cmap="gray")
    plt.show()

    return percentage

#print(prob6(4))