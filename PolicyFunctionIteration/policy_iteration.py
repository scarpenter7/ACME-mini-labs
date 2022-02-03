# policy_iteration.py
"""Volume 2: Policy Function Iteration.
<Name> Sam Carpenter
<Class>
<Date> 4/8/21
"""

import numpy as np
import gym
from gym import wrappers

# Intialize P for test example
#Left = 0
#Down = 1
#Right = 2
#Up= 3

P = {s : {a: [] for a in range(4)} for s in range(4)}
P[0][0] = [(0, 0, 0, False)]
P[0][1] = [(1, 2, -1, False)]
P[0][2] = [(1, 1, 0, False)]
P[0][3] = [(0, 0, 0, False)]
P[1][0] = [(1, 0, -1, False)]
P[1][1] = [(1, 3, 1, True)]
P[1][2] = [(0, 0, 0, False)]
P[1][3] = [(0, 0, 0, False)]
P[2][0] = [(0, 0, 0, False)]
P[2][1] = [(0, 0, 0, False)]
P[2][2] = [(1, 3, 1, True)]
P[2][3] = [(1, 0, 0, False)]
P[3][0] = [(0, 0, 0, True)]
P[3][1] = [(0, 0, 0, True)]
P[3][2] = [(0, 0, 0, True)]
P[3][3] = [(0, 0, 0, True)]
# probability, next state, reward, terminal y/n


# Problem 1
def value_iteration(P, nS ,nA, beta = 1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
       n (int): number of iterations
    """
    V = np.zeros(nS)
    numIters = 0

    # Iterate solve
    while numIters < maxiter:
        newV = V.copy()

        for s in range(nS):
            sa_vector = np.zeros(nA)
            for a in range(nA):
                for tuple_info in P[s][a]:
                    # tuple_info is a tuple of (probability, next state, reward, done)
                    p, s_, u, _ = tuple_info
                    # sums up the possible end states and rewards with given action
                    sa_vector[a] += (p * (u + beta * V[s_]))
            # add the max value to the value function
            newV[s] = np.max(sa_vector)
        numIters += 1
        if np.linalg.norm(newV - V, ord=2) < tol:
            V = newV
            return V, numIters
        V = newV

    return V, numIters

# Problem 2
def extract_policy(P, nS, nA, v, beta = 1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """
    # Initialize optimal policy
    optPolicy = np.zeros_like(v)

    # Loop thru  each starting state
    for startState in range(nS):
        rewards = []
        # Loop through each possible action for each starting state
        for a in range(nA):
            intermedReward = []
            for tuple in P[startState][a]:
                prob, nextState, reward, terminal = tuple
                intermedReward.append(prob*(reward + beta*v[nextState]))
            rewards.append(np.sum(intermedReward))
        # Take the argmax of the rewards array
        optPolicy[startState] = np.argmax(rewards)
    return optPolicy

# Problem 3
def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8):
    """Computes the value function for a policy using policy evaluation.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.

    Returns:
        v (ndarray): The discrete values for the true value function.
    """
    V = np.zeros(nS)

    # Iterate solve
    while True:
        newV = V.copy()
        # Iterate thru the states
        for s in range(nS):
            V[s] = np.sum([p * (u + beta * newV[s_]) for p, s_, u, _ in P[s][policy[s]]])
        if np.linalg.norm(newV - V, ord=2) < tol: # converged
            V = newV
            return V

# Problem 4
def policy_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
    	v (ndarray): The discrete values for the true value function
        policy (ndarray): which direction to move in each square.
        n (int): number of iterations
    """
    # Follow code in the book
    policy = np.zeros(nS)
    for k in range(maxiter):
        # Use previous functions
        v_k1 = compute_policy_v(P, nS, nA, policy, beta=beta,tol=tol)
        newPolicy = extract_policy(P, nS, nA, v_k1, beta=beta)

        # Check for convergence
        if np.linalg.norm(newPolicy - policy, ord=2) < tol:
            return v_k1, newPolicy, k + 1
        policy = newPolicy


# Problem 5 and 6
def frozen_lake(basic_case=True, M=1000, render=False):
    """ Finds the optimal policy to solve the FrozenLake problem

    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns.
    M (int): The number of times to run the simulation using problem 6.
    render (boolean): Whether to draw the environment.

    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The mean expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value function for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The mean expected value for following the policy iteration optimal policy.
    """

    if basic_case:
        env_name = 'FrozenLake-v0'
    else:
        env_name = 'FrozenLake8x8-v0'

    env = gym.make(env_name).env
    nS = env.nS
    nA = env.nA
    # Get the dictionary with all the states and actions
    dictionary_P = env.P
    v = value_iteration(dictionary_P, nS, nA)[0]
    viPolicy = extract_policy(dictionary_P, nS, nA, v)
    piValFunc, piPolicy, iters = policy_iteration(dictionary_P, nS, nA)

    vRewards = []
    piRewards = []

    for i in range(M):
        vRewards.append(run_simulation(env, viPolicy, render=render))
        piRewards.append(run_simulation(env, piPolicy, render=render))
    env.close()

    return viPolicy, sum(vRewards)/len(vRewards), piValFunc, piPolicy, sum(piRewards)/len(piRewards)


# Problem 6
def run_simulation(env, policy, render=True, beta = 1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.

    Parameters:
    env (gym environment): The gym environment.
    policy (ndarray): The policy used to simulate.
    beta float: The discount factor.
    render (boolean): Whether to draw the environment.

    Returns:
    total reward (float): Value of the total reward received under policy.
    """
    # Put environment in starting state
    obs = env.reset()
    # Take a step in the optimal direction and update variables
    obs, reward, done, _ = env.step(int(policy[obs]))
    steps = 1
    if render: env.render(mode = 'human')
    while not done:
        obs, reward, done, _ = env.step(int(policy[obs]))
        if render: env.render(mode = 'human')
        steps += 1
    return reward * beta**steps


# Tests

def prob1Test():
    print(value_iteration(P, 4, 4))

def prob2Test():
    v, iters = value_iteration(P, 4, 4)
    print(extract_policy(P, 4, 4, v))

def prob3Test():
    v, iters = value_iteration(P, 4, 4)
    policy = extract_policy(P, 4, 4, v)
    print(v)
    print(compute_policy_v(P, 4, 4, policy))

def prob4Test():
    print(policy_iteration(P, 4, 4))

def prob4Test2():
    P2 = {s: {a: [(0, 0, 0, False)] for a in range(4)} for s in range(6)}
    P2[0][2] = [(1, 1, .1, False)]
    P2[0][1] = [(1, 3, -1, False)]
    P2[1][0] = [(1, 0, -1, False)]
    P2[1][2] = [(1, 2, 0, False)]
    P2[1][1] = [(1, 4, -1, False)]
    P2[2][0] = [(1, 1, -1, False)]
    P2[2][1] = [(1, 5, 2, True)]
    P2[3][3] = [(1, 0, -1, False)]
    P2[3][2] = [(1, 4, -1, False)]
    P2[4][3] = [(1, 1, .1, False)]
    P2[4][0] = [(1, 3, -1, False)]
    P2[4][2] = [(1, 5, 2, True)]
    P2[5][1] = [(0, 5, 0, True)]
    for string in policy_iteration(P2, 6, 4):
        print(string)

def prob56Test1():
    result = frozen_lake()
    for r in result:
        print(r)

def prob56Test2():
    result = frozen_lake(basic_case=False, render=True)
    for r in result:
        print(r)



# prob1Test()
# prob2Test()
# prob3Test()
# prob4Test()
# prob4Test2()
# prob56Test1()
# prob56Test2()
