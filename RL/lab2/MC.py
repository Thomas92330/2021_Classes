import numpy as np
import random

# parameters
gamma = 0.5 # discounting rate
rewardSize = -1
gridSize = 4
alpha = 0.5 # (0,1] // stepSize
terminationStates = [[0,0], [gridSize-1, gridSize-1]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numIterations = 10000

# initialization
V_firstVisit_MC = np.zeros((gridSize, gridSize))
V_everyVisit_MC = np.zeros((gridSize, gridSize))
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]# utils


def generateRandomInitialState():
    return random.choice(states[1:-1])

def generateRandomNextAction():
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
    # TODO
    # pour chaque état, évaluer la value function
    for i in range(gridSize):
        for j in range(gridSize):
            initState = [i,j] # ligne, colonne
            N = 0
            S = 0
            N_every = 0
            S_every = 0
            # je regarde si j'ai fini ma partie
            if (list(initState) in terminationStates):
                continue #je joue 10000 parties pour calculer v[0,1]
            for k in range (numIterations): 
                currentState = initState
                # a chaque fois que je passe pour la premiere fois sur un état, j'incrémente counter (nb de passage)
                # je commence ma partie à initState
                N += 1
                N_every += 1 
                # tant que je ne suis pas dans un etat terminal, je joue ma partie
                while True:
                    # je prends une action random
                    action = generateRandomNextAction()
                    
                    # je joue le coup
                    reward, nextState = takeAction(currentState, action)
                    
                    # j'ajoute mon reward a la somme S
                    S+=(reward)
                    S_every+=(reward*N_every)
                    # je regarde si j'ai fini ma partie
                    if (list(currentState) in terminationStates):
                        break
                    # je mets a jour mon état
                    currentState=nextState
                    if (currentState == initState):
                        N_every+=1
            # calculer mes rewards jusqu'a la fin de la partie et les ajouter à S
            # a la fin des parties je fais la moyenne S/N 
            V_firstVisit_MC[i,j] = S/N
            V_everyVisit_MC[i,j] = S_every/N_every
    print(V_firstVisit_MC)
    print(V_everyVisit_MC)
                        