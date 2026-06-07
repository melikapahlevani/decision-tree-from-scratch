import pandas as pd
import numpy as np
from graphviz import Digraph
import math


class Node():
    # part1
    def __init__(self, feature=None, children=None, value=None, feature_value=None):
        self.feature = feature
        self.children = children
        # if it is leaf
        self.value = value
        # for part 4
        self.feature_value=feature_value


class DecisionTree():
    # part2
    def __init__(self, mode=None, max_Depth=float("inf"), min_Samples=2,
                 pruning_threshold=None, root=None):
        self.mode = mode
        self.max_Depth = max_Depth
        self.min_Samples = min_Samples
        self.pruning_threshold = pruning_threshold
        self.root = root

    def _create_Tree(self, X, Y, depth=0):
        num_Samples = len(Y)
        # if all in one class
        if (len(np.unique(Y))==1)or len(X.columns) == 0:
            return Node(value=self._calculate_Value(Y))
        
        if depth >= self.max_Depth or num_Samples < self.min_Samples or len(X.columns) == 0:
            return Node(value=self._calculate_Value(Y))
        
        best_Feature = self._get_best_Feature(X, Y)
        if best_Feature is None or best_Feature.feature is None :
                return Node(value=self._calculate_Value(Y))
        
        children = []
        fr = best_Feature.feature
        for c in X[fr].unique():
                xi = X[X[fr]==c].drop(columns=[fr])
                yi = Y[X[fr]==c]

                if len(yi)<self.min_Samples:
                    child = Node(value=self._calculate_Value(yi), feature_value=c)
                    # child = Node(value=self._calculate_Value(yi))
                else:
                    child = self._create_Tree(xi,yi,depth+1)
                    child.feature_value = c 
                children.append(child)
        return Node(feature=best_Feature.feature,children=children,value=self._calculate_Value(Y),feature_value=best_Feature.feature_value)


    # part3.3
    def _get_best_Feature(self, X, Y):
        bestf = Node()
        if self.mode == "gain":
            best = -1000
            for feature in X.columns:
                test = self._information_Gain(X[feature], X, Y)
                if test > best:
                    best = test
                    bestf.feature = feature
            return bestf
        elif self.mode == "gini":
            best = 0.5
            bestf = Node()
            for feature in X.columns:
                test = self._gini_Split(X[feature], X, Y)
                if test < best :
                    best = test
                    bestf.feature = feature
            return bestf

    # part3.1
    def _information_Gain(self, feature, X, Y):
        def entropy(y):
            en = 0
            classes = np.unique(y)
            for c in classes:
                pc = (np.sum(y == c) / len(y))
                if pc > 0 and pc < 1:
                    en += -(np.log2(pc) * pc)
                elif pc == 0:
                    en += 0
                elif pc == 1:
                    return 0
            return en

        rem = 0
        for value in feature.unique():
            rem += ((len(Y[feature == value])) / len(feature)) * (entropy(Y[feature == value]))
        return entropy(Y) - rem

    # part3.2
    def _gini_Split(self, feature, X, Y):
        def gini(y):
            gin = 0
            classes = np.unique(y)
            for c in classes:
                pc = np.sum(y == c) / len(y)
                gin += pc ** 2
            return 1 - gin

        ginall = 0
        for value in feature.unique():
            ginall += (len(Y[feature == value]) / len(feature)) * gini(Y[feature == value])
        return ginall

    # if all features done or we create child with no Y or max_depth
    def _calculate_Value(self, Y):
        return max(set(Y), key=list(Y).count)

    # part4
    def fit(self, X_Train, Y_Train):
        self.root = self._create_Tree(X_Train, Y_Train)

    def predict(self, X):
        def _move_Tree(sample, root):
            if root.children == None :
                return root.value
            # if root.value != None :
                # return root.value
            else:
                for c in root.children :
                    if sample[root.feature] == c.feature_value:
                       return _move_Tree(sample,c)
            return root.value
        # for s in X.columns:
        #     _move_Tree(s,self.root)
        preds = []
        for i in range(len(X)):
            sample = X.iloc[i]
            preds.append(_move_Tree(sample,self.root))
        return preds
    
