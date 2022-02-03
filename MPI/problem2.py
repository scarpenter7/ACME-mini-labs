# Problem 2
"""Pass a random NumPy array of shape (n,) from the root process to process 1,
where n is a command-line argument. Print the array and process number from
each process.

Usage:
    # This script must be run with 2 processes.
    $ mpiexec -n 2 python problem2.py 4
    Process 1: Before checking mailbox: vec=[ 0.  0.  0.  0.]
    Process 0: Sent: vec=[ 0.03162613  0.38340242  0.27480538  0.56390755]
    Process 1: Recieved: vec=[ 0.03162613  0.38340242  0.27480538  0.56390755]
"""
from mpi4py import MPI
import numpy as np
from sys import argv
# Pass in the first command line argument as n.
n = int(argv[1])
# print(n)
RANK = MPI.COMM_WORLD.Get_rank()

a = np.zeros(n)
if RANK == 0:
    # a[0] = np.array([random.random() for _ in range(n)])
    a = np.random.rand(n)  # This must be an array.
    MPI.COMM_WORLD.Send(a, dest=1)
    print("Process 0: Sent: vec=" + str(a))
elif RANK == 1:
    print("Process 1: Before checking mailbox: vec=" + str(a))
    MPI.COMM_WORLD.Recv(a, source=0)
    print("Process 1: Recieved: vec=" + str(a))

