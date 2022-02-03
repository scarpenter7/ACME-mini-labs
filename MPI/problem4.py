# Problem 4
"""The n-dimensional open unit ball is the set U_n = {x in R^n : ||x|| < 1}.
Estimate the volume of U_n by making N draws on each available process except
for the root process. Have the root process print the volume estimate.

Command line arguments:
    n (int): the dimension of the unit ball.
    N (int): the number of random draws to make on each process but the root.

Usage:
    # Estimate the volume of U_2 (the unit circle) with 2000 draws per process.
    $ mpiexec -n 4 python problem4.py 2 2000
    Volume of 2-D unit ball: 3.13266666667      # Results will vary slightly.
"""

from mpi4py import MPI
import numpy as np
from sys import argv
import random
from scipy import linalg as la

Comm = MPI.COMM_WORLD
num_processes = Comm.Get_size()

# Take in cmd args
n = int(argv[1])
N = int(argv[2])

vol = 2**n

RANK = MPI.COMM_WORLD.Get_rank()

area = np.zeros(1)

if RANK != 0:
    # sample N points, compute the area, send it
    draws = np.random.uniform(-1, 1, (n,N))
    points = np.reshape(draws, (N, n))
    lengths = la.norm(points, axis=1)
    num_within = np.count_nonzero(lengths < 1)
    area[0] = vol * (num_within / N)
    Comm.Send(area, dest=0)
else: # RANK = 0
    areas = []
    for i in range(1, num_processes):
        Comm.Recv(area, source=i)
        areas.append(area[0])
    # print avg of each area for each process
    print(np.mean(areas))

