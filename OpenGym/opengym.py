# opengym.py
"""Volume 2: Open Gym
<Name> Sam Carpenter
<Class>
<Date> 2/18
"""

import gym
import numpy as np
from IPython.display import clear_output
import random

def find_qvalues(env,alpha=.1,gamma=.6,epsilon=.1):
    """
    Use the Q-learningx algorithm to find qvalues.

    Parameters:
        env (str): environment name
        alpha (float): learning rate
        gamma (float): discount factor
        epsilon (float): maximum value

    Returns:
        q_table (ndarray nxm)
    """
    # Make environment
    env = gym.make(env)
    # Make Q-table
    q_table = np.zeros((env.observation_space.n,env.action_space.n))

    # Train
    for i in range(1,100001):
        # Reset state
        state = env.reset()

        epochs, penalties, reward, = 0,0,0
        done = False

        while not done:
            # Accept based on alpha
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Take action
            next_state, reward, done, info = env.step(action)

            # Calculate new qvalue
            old_value = q_table[state,action]
            next_max = np.max(q_table[next_state])

            new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            # Check if penalty is made
            if reward == -10:
                penalties += 1

            # Get next observation
            state = next_state
            epochs += 1

        # Print episode number
        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.")
    return q_table

# Problem 1
def random_blackjack(n):
    """
    Play a random game of Blackjack. Determine the
    percentage the player wins out of n times.

    Parameters:
        n (int): number of iterations

    Returns:
        percent (float): percentage that the player
                         wins
    """
    # The
    # observation (rst entry in the tuple returned by step) is a tuple containing the total sum of
    # the players hand, the rst card of the computer's hand, and whether the player has an ace.
    # The reward (second entry in the tuple returned by step) is 1 if the player wins, -1 if the player
    # loses, and 0 if there is a draw.
    wins = 0
    for _ in range(n):
        env = gym.make("Blackjack-v0")
        observation = env.reset()
        # print(observation)

        result = env.step(env.action_space.sample())
        # print(result)
        while not result[2]:
            result = env.step(env.action_space.sample())
            # print(result)
        if result[1] == 1:
            wins += 1
    return wins / n

# Problem 2
def blackjack(n=11):
    """
    Play blackjack with naive algorithm.

    Parameters:
        n (int): maximum accepted player hand

    Return:
        percent (float): percentage of 10000 iterations
                         that the player wins
    """
    wins = 0
    for _ in range(10000):
        env = gym.make("Blackjack-v0")
        intialObs = env.reset()
        currentHand = intialObs[0]
        #print(intialObs)
        done = False
        while not done and currentHand < n:
            obs, reward, done, info = env.step(1)
            currentHand = obs[0]
            #print(obs)
            #print(reward)
            #print(done)
            #print(info)
            #print()
            if done:
                break
        obs, reward, done, info = env.step(0)
        if reward == 1:
            wins += 1

    return wins / 10000

# Problem 3
def cartpole():
    """
    Solve CartPole-v0 by checking the velocity
    of the tip of the pole

    Return:
        iterations (integer): number of steps or iterations
                              to solve the environment
    """
    env = gym.make("CartPole-v0" )
    try:
        initial = env.reset()
        velocity = initial[3]
        steps = 0
        done = False
        while not done:
            env.render()
            # Always move in the direction of the velocity of the pole
            if velocity > 0:
                obs, reward, done, info = env.step(1)
            else:
                obs, reward, done, info = env.step(0)
            # Keep track of velocity as you go
            velocity = obs[3]
            steps += 1
            if done:
                break
    finally:
        env.close()
    return steps

# Problem 4
def car():
    """
    Solve MountainCar-v0 by checking the position
    of the car.

    Return:
        iterations (integer): number of steps or iterations
                              to solve the environment
    """
    env = gym.make("MountainCar-v0")
    try:
        initial = env.reset()
        velocity = initial[1]
        steps = 0
        done = False
        while not done:
            env.render()
            # Keep going back and forth until you make it
            if velocity > 0:
                obs, reward, done, info = env.step(2)
            else:
                obs, reward, done, info = env.step(0)
            # Check velocity
            velocity = obs[1]
            steps += 1
            if done:
                break
    finally:
        env.close()
    return steps

# Problem 5
def taxi(q_table):
    """
    Compare naive and q-learning algorithms.

    Parameters:
        q_table (ndarray nxm): table of qvalues

    Returns:
        naive (flaot): mean reward of naive algorithm
                       of 10000 runs
        q_reward (float): mean reward of Q-learning algorithm
                          of 10000 runs
    """
    env = gym.make("Taxi-v3")
    rewards = []
    for _ in range(10000):
        try:
            initial = env.reset()
            done = False
            totalReward = 0

            # Random actions and keep track of reward.
            while not done:
                obs, reward, done, info = env.step(env.action_space.sample())
                totalReward += reward
                if done:
                    break
        finally:
            env.close()
        rewards.append(totalReward)
    # Calculate avg reward
    randomReward = sum(rewards) / len(rewards)

    # See helper function for Q table version
    q_tableReward = qReward(q_table)

    return randomReward, q_tableReward

def qReward(q_table):
    rewards = []
    for _ in range(10000):
        env = gym.make("Taxi-v3")
        try:
            obs = env.reset()
            done = False
            totalReward = 0
            # Make the best decision based on Q table
            while not done:
                obs, reward, done, info = env.step(np.argmax(q_table[obs, :]))
                totalReward += reward
                if done:
                    break
            rewards.append(totalReward)
        finally:
            env.close()
    # calculate avg reward
    return sum(rewards) / len(rewards)

# Tests
def prob1Test():
    print(random_blackjack(30))

def prob2Test():
    print(blackjack())
    print(blackjack(15))
    print(blackjack(20))
    print(blackjack(1000))

def prob3Test():
    print(cartpole())

def prob4Test():
    print(car())

def prob5Test():
    table = find_qvalues("Taxi-v3")
    print(taxi(table))

# prob1Test()
# prob2Test()
# prob3Test()
# prob4Test()
# prob5Test()



