# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 09:19:41 2020

@author: yoann
"""

import numpy as np
import random

# parameters
gamma = 0.5 # discounting rate
rewardSize = -1
gridSize = 4
alpha = 0.5 # (0,1] // stepSize
terminationStates = [[0,0], [gridSize-1, gridSize-1]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numIterations = 1000

# initialization
V = np.zeros((gridSize, gridSize))
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

# utils
def generateInitialState():
    return random.choice(states[1:-1])

def generateNextAction():
    return random.choice(actions)

def takeAction(state, action):
    if list(state) in terminationStates:
        return 0, None
    finalState = np.array(state)+np.array(action)
    # if robot crosses wall
    if -1 in list(finalState) or gridSize in list(finalState):
        finalState = state
    return rewardSize, list(finalState)       


if __name__ == "__main__":
    
    #TODO
    initial_state = generateInitialState()
    for i in range (numIterations):
        number_of_move = 1
        G = 0
        state_list = [initial_state]
        current_state = initial_state
        while current_state not in terminationStates :
            action = generateNextAction()
            reward , next_state = takeAction(current_state, action)
            if next_state not in state_list : 
                state_list.append(next_state)
                number_of_move +=  1
                G += reward 
                current_state = next_state
        V += G/number_of_move
        print(V)
    
