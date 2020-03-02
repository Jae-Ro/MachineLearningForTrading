"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch

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
"""

import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import math


class DTLearner(object):

    def __init__(self, verbose=False, leaf_size=1):
        self.verbose = verbose
        self.leaf_size = leaf_size

        self.dt = None
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        return 'jro32'  # replace tb34 with your Georgia Tech username

    def build_tree(self, data):
        dataX = data[:, :-1]
        dataY = data[:, -1]
        dataY = dataY.reshape(dataY.shape[0], 1)

        # IF the number of rows in dataX is <=1
        if (dataX.shape[0] <= self.leaf_size or np.unique(dataY).shape[0] == 1):
            leaf = [[-1, np.median(dataY), np.nan, np.nan]]
            return np.array(leaf)

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
            return np.array([[-1, np.median(dataY), np.nan, np.nan]])

        lefttree = self.build_tree(dataLeft)
        righttree = self.build_tree(dataRight)

        root = np.array([[i, SplitVal, 1, lefttree.shape[0]+1]])

        return np.concatenate((root, lefttree, righttree))

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        X = np.array(dataX)
        Y = np.array(dataY)
        Y = Y.reshape(Y.shape[0], 1)

        data = np.hstack((X, Y))
        self.dt = self.build_tree(data)
        print(self.dt)

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        predictions = []
        # print("NUMBER OF QUERY POINTS: ", len(points))

        for i in range(len(points)):
            pred = self.traverse_tree(self.dt, points[i, :])
            predictions.append(pred)

        return predictions

    def traverse_tree(self, tree, row):
        # Base Case: where feature index is not 0->n, but is a leaf(denoted by -1)
        i = 0
        while True:
            if (tree[i, :][0] < 0):
                return tree[i, :][1]

            feature_index = int(tree[i, :][0])
            # go down left tree
            if(row[feature_index] <= tree[i, :][1]):
                i += 1
            # go down right tree
            elif(row[feature_index] > tree[i, :][1]):
                i += int(tree[i, :][3])


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
    df = pd.read_csv('Istanbul.csv')
    y_values = df.iloc[:, -1]
    x_values = df.iloc[:, 1:-1]

    # data = np.array([[0.03575371,  0.03837619, -0.00467931,  0.00219342,  0.00389438,
    #                   0.,  0.03119023,  0.01269804,  0.02852446],
    #                  [0.02542587,  0.03181274,  0.00778674,  0.00845534,  0.01286561,
    #                   0.00416245,  0.01891958,  0.01134065,  0.00877264],
    #                  [-0.02886173, -0.02635297, -0.03046913, -0.01783306, -0.02873459,
    #                   0.01729293, -0.03589858, -0.0170728, -0.02001541],
    #                  [-0.06220808, -0.0847159,  0.00339136, -0.01172628, -0.000466,
    #                   -0.04006131,  0.02828315, -0.00556096, -0.01942378],
    #                  [0.00985991,  0.00965811, -0.02153321, -0.01987275, -0.01270972,
    #                   -0.0044735, -0.00976388, -0.01098863, -0.00780221]])

    x_values = np.array(x_values)
    y_values = np.array(y_values)
    y_values = y_values.reshape(y_values.shape[0], 1)
    data = np.hstack((x_values, y_values))
    print(data.shape)

    x = data[:10, :-1]
    y = data[:10, -1]
    y = y.reshape(y.shape[0], 1)

    print(x.shape, y.shape)

    dt = DTLearner()
    dt.addEvidence(x, y)

    print(dt.query(x[:50]))
