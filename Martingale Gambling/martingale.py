"""Assess a betting strategy.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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
		   	  			  	 		  		  		    	 		 		   		 		  
"""

import numpy as np
import matplotlib.pyplot as plt


def author():
    return 'jro32'  # replace tb34 with your Georgia Tech username.


def gtid():
    return 903450684  # replace with your GT ID number


def get_spin_result(win_prob):
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def test_code():
    win_prob = 18.0/38.0  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    print(get_spin_result(win_prob))  # test the roulette spin
    experiment_one(win_prob)
    experiment_two(win_prob)


def experiment_one(win_prob):
    # FIGURE 1
    plt.figure(1)
    plt.xlim((0, 300))
    plt.ylim((-256, 100))

    for i in range(10):
        episode_winnings = simple_simulator(win_prob)
        # exp1_winnings.append(episode_winnings)
        # print(episode_winnings[:5])
        trial_label = "Trial" + str(i+1)
        plt.plot(episode_winnings, label=trial_label)
    plt.legend()
    plt.savefig("Figure1.png")

    # FIGURE 2
    plt.figure(2)
    plt.xlim((0, 300))
    plt.ylim((-256, 100))
    exp1_winnings_fig2 = []
    for i in range(1000):
        episode_winnings = simple_simulator(win_prob)
        exp1_winnings_fig2.append(episode_winnings)

    exp1_winnings_fig2 = np.array(exp1_winnings_fig2)

    mean_values = np.mean(exp1_winnings_fig2, axis=0)
    std_values = np.std(exp1_winnings_fig2, axis=0)

    plt.plot(mean_values, label="Means Across 1000 iterations")
    plt.plot(mean_values + std_values, label="Means + std")
    plt.plot(mean_values - std_values, label="Means - std")
    plt.legend()
    plt.savefig("Figure2.png")

    # FIGURE 3
    plt.figure(3)
    plt.xlim((0, 300))
    plt.ylim((-256, 100))

    median_values = np.median(exp1_winnings_fig2, axis=0)
    std_values = np.std(exp1_winnings_fig2, axis=0)

    plt.plot(median_values, label="Medians Across 1000 iterations - Simple")
    plt.plot(median_values + std_values, label="Medians + std")
    plt.plot(median_values - std_values, label="Medians - std")
    plt.legend()
    plt.savefig("Figure3.png")

    plt.figure(3)
    plt.xlim((0, 300))
    plt.ylim((-256, 100))

    median_values = np.median(exp1_winnings_fig2, axis=0)
    std_values = np.std(exp1_winnings_fig2, axis=0)

    plt.plot(median_values, label="Medians Across 1000 iterations - Simple")
    plt.plot(median_values + std_values, label="Medians + std")
    plt.plot(median_values - std_values, label="Medians - std")
    plt.legend()
    plt.savefig("Figure3.png")

    # # EXPERIMENT 1 Standard Deviations
    # plt.figure(6)
    # # plt.xlim((0, 300))
    # # plt.ylim((-256, 100))

    # std_values = np.std(exp1_winnings_fig2, axis=0)
    # plt.plot(std_values, label="Exp1 Standard Deviation")
    # plt.legend()
    # plt.savefig("Figure6.png")


def experiment_two(win_prob):
    # FIGURE 4
    plt.figure(4)
    plt.xlim((0, 1000))
    plt.ylim((-256, 100))
    exp2_winnings_fig4 = []

    for i in range(1000):
        episode_winnings = realistic_simulator(win_prob)
        exp2_winnings_fig4.append(episode_winnings)

    exp2_winnings_fig4 = np.array(exp2_winnings_fig4)
    mean_values_exp2 = np.mean(exp2_winnings_fig4, axis=0)
    std_values_exp2 = np.std(exp2_winnings_fig4, axis=0)
    plt.plot(mean_values_exp2, label="Means Across 1000 iterations - Realistic")
    plt.plot(mean_values_exp2 + std_values_exp2, label="Means + std")
    plt.plot(mean_values_exp2 - std_values_exp2, label="Means - std")
    plt.legend()
    plt.savefig("Figure4.png")

    # FIGURE 5
    plt.figure(5)
    plt.xlim((0, 300))
    plt.ylim((-256, 100))
    median_values_exp2 = np.median(exp2_winnings_fig4, axis=0)
    plt.plot(median_values_exp2,
             label="Medians Across 1000 iterations - Realistic")
    plt.plot(median_values_exp2 + std_values_exp2, label="Medians + std")
    plt.plot(median_values_exp2 - std_values_exp2, label="Medians - std")
    plt.legend()
    plt.savefig("Figure5.png")

    # # EXPERIMENT 2 Standard Deviations
    # plt.figure(7)
    # # plt.xlim((0, 300))
    # # plt.ylim((-256, 100))

    # std_values = np.std(exp2_winnings_fig4, axis=0)
    # plt.plot(std_values, label="Exp2 Standard Deviation")
    # plt.legend()
    # plt.savefig("Figure7.png")


def simple_simulator(win_prob):

    # add your code here to implement the experiments
    bet_black = False
    episode_winnings = 0
    counter = 0
    winnings = []
    winnings.append(0)
    max_default = 0
    while (counter < 1000):
        won = False
        bet_amount = 1
        while (not won and counter < 1000):
            bet_black = True
            won = get_spin_result(win_prob)
            if (won == True):

                episode_winnings = episode_winnings + bet_amount
                if (episode_winnings >= 80):
                    for x in range(counter, 1000):
                        winnings.append(80)
                    counter = 1000
                else:
                    winnings.append(episode_winnings)
                    counter += 1
            else:

                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount*2
                winnings.append(episode_winnings)
                counter += 1

    winnings = np.array(winnings)

    return winnings


def realistic_simulator(win_prob):
    wallet = 256
    bet_black = False
    episode_winnings = 0
    counter = 0
    winnings = []
    winnings.append(0)
    max_default = 0

    while (counter < 1000):
        won = False
        bet_amount = 1
        while (not won and counter < 1000):
            bet_black = True
            won = get_spin_result(win_prob)
            if (won == True):
                episode_winnings = episode_winnings + bet_amount
                wallet += bet_amount

                if (episode_winnings >= 80):
                    # print(1000-counter)
                    for x in range(counter, 1000):
                        winnings.append(80)
                    counter = 1000
                else:
                    winnings.append(episode_winnings)
                    counter += 1
            else:
                episode_winnings = episode_winnings - bet_amount
                wallet -= bet_amount
                bet_amount = bet_amount*2
                if (bet_amount > wallet):
                    bet_amount = wallet

                if (wallet <= 0):
                    for y in range(counter, 1000):
                        winnings.append(-256)
                    counter = 1000
                else:
                    winnings.append(episode_winnings)
                    counter += 1

    winnings = np.array(winnings)
    # print(winnings.shape)
    return winnings


if __name__ == "__main__":
    test_code()
