# Problem 3
"""In each process, generate a random number, then send that number to the
process with the next highest rank (the last process should send to the root).
Print what each process starts with and what each process receives.

Usage:
    $ mpiexec -n 2 python problem3.py
    Process 1 started with [ 0.79711384]        # Values and order will vary.
    Process 1 received [ 0.54029085]
    Process 0 started with [ 0.54029085]
    Process 0 received [ 0.79711384]

    $ mpiexec -n 3 python problem3.py
    Process 2 started with [ 0.99893055]
    Process 0 started with [ 0.6304739]
    Process 1 started with [ 0.28834079]
    Process 1 received [ 0.6304739]
    Process 2 received [ 0.28834079]
    Process 0 received [ 0.99893055]
"""
from mpi4py import MPI
import numpy as np
from sys import argv
import random

RANK = MPI.COMM_WORLD.Get_rank()
Comm = MPI.COMM_WORLD
num_processes = Comm.Get_size()

num = np.array([random.random()])

# Receive the number
Comm.Recv(num, source=(RANK - 1) % num_processes)
print("Process " + str(RANK) +  " received " + str(num))

# Generate the number
print("Process " + str(RANK) +  " started with " + str(num))
Comm.Send(num, dest=(RANK + 1) % num_processes)




