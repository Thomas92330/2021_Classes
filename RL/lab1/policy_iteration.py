# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 08:52:02 2020

@author: yoann
"""

import numpy as np

#variables
GAMMA = 1
STEP_REWARD = -1
GRID_SIZE = 4
FINAL_STATES = [[0,0], [GRID_SIZE-1, GRID_SIZE-1]]
ACTIONS = [[-1, 0], [1, 0], [0, 1], [0, -1]]

def actionRewardFunction(initialPosition, action):
    
    #case where we are at a final state
    if initialPosition in FINAL_STATES:
        return initialPosition, 0
    
    finalPosition = np.array(initialPosition) + np.array(action)
    if -1 in finalPosition or GRID_SIZE in finalPosition: 
        finalPosition = initialPosition
        
    return finalPosition, STEP_REWARD



if __name__ == "__main__":
    
    #initialize value functions to 0
    value_function = np.zeros((GRID_SIZE,GRID_SIZE))
    #Iterate over states to update value functions
    previous_value_fonction = value_function
    for i in range (GRID_SIZE):
        for j in range (GRID_SIZE):
            for act in ACTIONS:
                new_value_function , reward = actionRewardFunction(previous_value_function, act)
                
    #Repeat the process
    
    #Define stop condition is improvement for all value function is less than 10^(-2)
    
    #Deduce Policy map
    
    #Create an agent which play with the environment
