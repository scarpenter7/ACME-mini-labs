from mpi4py import MPI

# Problem 1
"""Print 'Hello from process n' from processes with an even rank and
print 'Goodbye from process n' from processes with an odd rank (where
n is the rank).

Usage:
    $ mpiexec -n 4 python problem1.py
    Goodbye from process 3                  # Order of outputs will vary.
    Hello from process 0
    Goodbye from process 1
    Hello from process 2

    # python problem1.py
    Hello from process 0
"""
RANK = MPI.COMM_WORLD.Get_rank()

a = 2
b = 3
if RANK % 2 == 0:
    print("Hello from process " + str(RANK))
else :
    print("Goodbye from process " + str(RANK))


