"""  		   	  			  	 		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		   	  			  	 		  		  		    	 		 		   		 		  
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
GT User ID: tb34 (replace with your User ID)  		   	  			  	 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		   	  			  	 		  		  		    	 		 		   		 		  
"""

import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import math


class Node:

    def __init__(self, feature_index, val, left_branch=None, right_branch=None):
        self.feature_index = feature_index
        self.val = val
        self.left_branch = left_branch
        self.right_branch = right_branch


class DTLearner(object):

    def __init__(self, verbose=False, leaf_size=1):
        self.verbose = verbose
        self.leaf_size = leaf_size

        self.dt = None

    def build_tree(self, data):
        dataX = data[:, :-1]
        dataY = data[:, -1]
        dataY = dataY.reshape(dataY.shape[0], 1)

        # IF the number of rows in dataX is <=1
        if (dataX.shape[0] <= self.leaf_size):
            return Node(-1, np.median(dataY))

        if(np.unique(dataY).shape[0] == 1):
            return Node(-1, np.median(dataY))

        else:
            # Determine	best feature i to	split	on
            number_features = dataX.shape[1]-1
            correlationArr = []
            for j in range(number_features):
                feature = dataX[:, j]
                results = dataY[:, -1]

                corr = abs(np.corrcoef(feature, results)[0][1])
                if(math.isnan(corr)):
                    corr = 0

                correlationArr.append(corr)

            bestFeatureIndex = np.argmax(correlationArr)
            i = bestFeatureIndex
            SplitVal = np.median(data[:, i])

            dataLeft = data[data[:, i] <= SplitVal]
            dataRight = data[data[:, i] > SplitVal]

            if(dataLeft.size == 0 or dataRight.size == 0):
                return Node(-1, np.median(dataY))

            lefttree = self.build_tree(dataLeft)
            righttree = self.build_tree(dataRight)

            return Node(i, SplitVal, lefttree, righttree)

    def addEvidence(self, dataX, dataY):

        X = np.array(dataX)
        Y = np.array(dataY)
        Y = Y.reshape(Y.shape[0], 1)

        data = np.hstack((X, Y))
        self.dt = self.build_tree(data)

    def query(self, points):

        predictions = []
        for i in range(len(points)):
            pred = self.traverse_tree(self.dt, points[i, :])
            predictions.append(pred)

        return predictions

    def traverse_tree(self, tree, row):
        # Base Case: where feature index is not 0->n, but is a leaf(denoted by -1)
        if (tree.feature_index < 0):
            return tree.val

        feature_index = tree.feature_index
        feature_val = row[feature_index]

        if (feature_val <= tree.val):
            prediction = self.traverse_tree(tree.left_branch, row)
        elif (feature_val > tree.val):
            prediction = self.traverse_tree(tree.right_branch, row)

        return prediction


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
