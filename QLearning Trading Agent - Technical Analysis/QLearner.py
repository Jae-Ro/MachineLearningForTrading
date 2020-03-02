"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: jro32
GT ID: 903450684
"""

import numpy as np
import random as rand


class QLearner(object):

    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 dyna=0,
                 verbose=False):

        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.s = 0
        self.a = 0

        self.sim_arr = []

        # Initializing the Q-Table
        self.Q = np.zeros(shape=(self.num_states, self.num_actions))

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        # setting state
        self.s = s

        exploit_prob = np.random.uniform(0, 1)
        if exploit_prob >= self.rar:
            # best action to take given that we are in this new state s' (called s in this function)
            action = np.argmax(self.Q[s, :])
        else:
            action = rand.randint(0, self.num_actions-1)

        if self.verbose:
            print(f"s = {s}, a = {action}")

        # setting action
        self.a = action

        return action

    def query(self, s_prime, r):
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @param r: The ne state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """

        if (self.dyna > 0):
            # DynaQ
            self.dynaQ(self.s, self.a, s_prime, r)
            # Exploration vs Exploitation
            exploit_prob = np.random.uniform(0, 1)
            if exploit_prob >= self.rar:
                action = np.argmax(self.Q[s_prime, :])
            else:
                action = rand.randint(0, self.num_actions-1)

        else:
            # Updating the Q-table
            self.updateTable(self.s, self.a, s_prime, r)

            # Exploration vs Exploitation
            exploit_prob = np.random.uniform(0, 1)
            if exploit_prob >= self.rar:
                # Finding best next action to take after going to state s'
                action = np.argmax(self.Q[s_prime, :])
            else:
                action = rand.randint(0, self.num_actions-1)

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")

        self.s = s_prime
        self.a = action
        self.rar = self.rar * self.radr

        return action

    def dynaQ(self, s, a, s_prime, r):
        simulation = [s, a, s_prime, r]
        self.sim_arr.append(simulation)

        for i in range(self.dyna):
            [s_dyna, a_dyna, s_dyna_prime, r_dyna] = rand.choice(self.sim_arr)
            # array_sim = np.array(self.sim_arr)
            self.updateTable(s_dyna, a_dyna, s_dyna_prime, r_dyna)

    def updateTable(self, s, a, s_prime, r):
        # Updating the Q-table at row s and column r and return action
        curr_Q = self.Q[s, a]
        learning_rate = self.alpha
        reward = r
        discount_factor = self.gamma
        best_next_action = np.argmax(self.Q[s_prime, :])
        optimal_future_estimate_Q = self.Q[s_prime, best_next_action]
        learned_value = reward + discount_factor * optimal_future_estimate_Q

        new_Q = (1-learning_rate) * curr_Q + learning_rate * learned_value

        self.Q[s, a] = new_Q

        # Simplified Version: (though this is not the temporal difference version of the equation)
        # self.Q[s,a] = (1-self.alpha) * self.Q[s,a] + self.alpha * (r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime, :])])

    def author(self):
        return 'jro32'


def author():
    return 'jro32'


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
