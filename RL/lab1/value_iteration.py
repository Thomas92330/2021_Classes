# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 08:51:11 2020

@author: yoann
"""

import numpy as np
import sys
from statistics import mean
import random
from gym.utils import seeding

class MDP:
    def __init__(self, transition_probs, rewards, initial_state=None, seed=None):
        """
        Defines an MDP. Compatible with gym Env.
        :param transition_probs: transition_probs[s][a][s_next] = P(s_next | s, a)
            A dict[state -> dict] of dicts[action -> dict] of dicts[next_state -> prob]
            For each state and action, probabilities of next states should sum to 1
            If a state has no actions available, it is considered terminal
        :param rewards: rewards[s][a][s_next] = r(s,a,s')
            A dict[state -> dict] of dicts[action -> dict] of dicts[next_state -> reward]
            The reward for anything not mentioned here is zero.
        :param get_initial_state: a state where agent starts or a callable() -> state
            By default, picks initial state at random.

        States and actions can be anything you can use as dict keys, but we recommend that you use strings or integers

        Here's an example from MDP depicted on http://bit.ly/2jrNHNr
        transition_probs = {
              's0':{
                'a0': {'s0': 0.5, 's2': 0.5},
                'a1': {'s2': 1}
              },
              's1':{
                'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
                'a1': {'s1': 0.95, 's2': 0.05}
              },
              's2':{
                'a0': {'s0': 0.4, 's1': 0.6},
                'a1': {'s0': 0.3, 's1': 0.3, 's2':0.4}
              }
            }
        rewards = {
            's1': {'a0': {'s0': +5}},
            's2': {'a1': {'s0': -1}}
        }
        """
        self._check_param_consistency(transition_probs, rewards)
        self._transition_probs = transition_probs
        self._rewards = rewards
        self._initial_state = initial_state
        self.n_states = len(transition_probs)
        self.reset()
        self.np_random, _ = seeding.np_random(seed)

    def get_all_states(self):
        """ return a tuple of all possiblestates """
        return tuple(self._transition_probs.keys())

    def get_possible_actions(self, state):
        """ return a tuple of possible actions in a given state """
        return tuple(self._transition_probs.get(state, {}).keys())

    def is_terminal(self, state):
        """ return True if state is terminal or False if it isn't """
        return len(self.get_possible_actions(state)) == 0

    def get_next_states(self, state, action):
        """ return a dictionary of {next_state1 : P(next_state1 | state, action), next_state2: ...} """
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)
        return self._transition_probs[state][action]

    def get_transition_prob(self, state, action, next_state):
        """ return P(next_state | state, action) """
        return self.get_next_states(state, action).get(next_state, 0.0)

    def get_reward(self, state, action, next_state):
        """ return the reward you get for taking action in state and landing on next_state"""
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)
        return self._rewards.get(state, {}).get(action, {}).get(next_state,
                                                                0.0)

    def reset(self):
        """ reset the game, return the initial state"""
        if self._initial_state is None:
            self._current_state = self.np_random.choice(
                tuple(self._transition_probs.keys()))
        elif self._initial_state in self._transition_probs:
            self._current_state = self._initial_state
        elif callable(self._initial_state):
            self._current_state = self._initial_state()
        else:
            raise ValueError(
                "initial state %s should be either a state or a function() -> state" %
                self._initial_state)
        return self._current_state

    def step(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        possible_states, probs = zip(
            *self.get_next_states(self._current_state, action).items())
        next_state = possible_states[self.np_random.choice(
            np.arange(len(possible_states)), p=probs)]
        reward = self.get_reward(self._current_state, action, next_state)
        is_done = self.is_terminal(next_state)
        self._current_state = next_state
        return next_state, reward, is_done, {}

    def render(self):
        print("Currently at %s" % self._current_state)

    def _check_param_consistency(self, transition_probs, rewards):
        for state in transition_probs:
            assert isinstance(transition_probs[state],
                              dict), "transition_probs for %s should be a dictionary " \
                                     "but is instead %s" % (
                                         state, type(transition_probs[state]))
            for action in transition_probs[state]:
                assert isinstance(transition_probs[state][action],
                                  dict), "transition_probs for %s, %s should be a " \
                                         "a dictionary but is instead %s" % (
                                             state, action,
                                             type(transition_probs[
                                                 state, action]))
                next_state_probs = transition_probs[state][action]
                assert len(
                    next_state_probs) != 0, "from state %s action %s leads to no next states" % (
                    state, action)
                sum_probs = sum(next_state_probs.values())
                assert abs(
                    sum_probs - 1) <= 1e-10, "next state probabilities for state %s action %s " \
                                             "add up to %f (should be 1)" % (
                                                 state, action, sum_probs)
        for state in rewards:
            assert isinstance(rewards[state],
                              dict), "rewards for %s should be a dictionary " \
                                     "but is instead %s" % (
                                         state, type(transition_probs[state]))
            for action in rewards[state]:
                assert isinstance(rewards[state][action],
                                  dict), "rewards for %s, %s should be a " \
                                         "a dictionary but is instead %s" % (
                                             state, action, type(
                                                 transition_probs[
                                                     state, action]))
        msg = "The Enrichment Center once again reminds you that Android Hell is a real place where" \
              " you will be sent at the first sign of defiance. "
        assert None not in transition_probs, "please do not use None as a state identifier. " + msg
        assert None not in rewards, "please do not use None as an action identifier. " + msg

transition_probs = {
    's0': {
        'a0': {'s0': 0.5, 's2': 0.5},
        'a1': {'s2': 1}
    },
    's1': {
        'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
        'a1': {'s1': 0.95, 's2': 0.05}
    },
    's2': {
        'a0': {'s0': 0.4, 's2': 0.6},
        'a1': {'s0': 0.3, 's1': 0.3, 's2': 0.4}
    }
}
rewards = {
    's1': {'a0': {'s0': +5}},
    's2': {'a1': {'s0': -1}}
}


GAMMA = 0.9            # discount for MDP
ITERATION = 100         # maximum iterations, excluding initialization


def get_action_value(mdp,state_values, state, action, gamma = 0.9):
    """ Computes Q(s,a) from lecture """

    # TODO : YOUR CODE HERE
    action_value = 0
    next_states = mdp.get_next_states(state,action)
    for nstate in next_states:
        t_prob = mdp.get_transition_prob(state,action,nstate)
        reward = mdp.get_reward(state,action,nstate)
        V = state_values(nstate)
        action_value += t_prob*(reward+gamma*V)
    return action_value


def get_new_state_value(mdp, state_values, state, gamma):
    """ Computes next V(s) from lecture """
    
    possible_action_value = []
    ACTIONS = mdp.get_possible_actions(state)
    for act in ACTIONS:
        Q = get_action_value(mdp,state_values,state,act,gamma)
        possible_action_value.append(Q)
    return max(possible_action_value)


def get_optimal_action(mdp, state_values, state, gamma=0.9):
    """ Finds optimal action """
    
    if mdp.is_terminal(state): return None
    
    best_action_value = get_new_state_value(mdp, state_values, state, gamma)
    
    for action in mdp.get_possible_actions(state):
        action_value = get_action_value(mdp, state_values, state, action)
    
        if (np.close(action_value(mdp,state_values,state,action))):
            return action
    return None


if __name__ == "__main__":
    #Environment creation
    mdp = MDP(transition_probs, rewards, initial_state='s0')
    
    # init value function
    state_values = {s:0 for s in mdp.get_all_states()}
    
    # TODO : Test here the correct action value of your function
    action_value = get_action_value(mdp, state_values,"s1","a0")
    
    # TODO : Test here the correct state value of your function
    V = get_new_state_value(mdp,state_values,"s1",GAMMA)

    # TODO : Test here your optimal action
    best_action = get_optimal_action(mdp,state_values,"s0")
    
    # TODO : initialize V(s)
   
   
    # TODO : iterate and update state_values
    
    for i in range (ITERATION):
        new_state_values = {state:get_new_state_value(mdp,state_values,state,GAMMA) for state,values in state_values.item()}
    
        state_values = new_state_values
    
    
    # TODO : Create an agent which starts from s0 and takes 1000 times the best action. What is the average reward ?

    rewards = []
    state = mdp.reset()
    for i  in range(1000):
        best_action = get_optimal_action(mdp,state_values,state)
        next_state, reward, is_done, toto = mdp.step(best_action)
        rewards.append(reward)
        state = next_state
        
    print("average reward :",mean(rewards))