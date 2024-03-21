# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:34:56 2022

@author: Asus
"""

import numpy as np
import pandas as pd
#from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
#import matplotlib.pyplot as plt
from scipy.linalg import svd

# Load the RNA csv file and its label file
filename = 'data.csv'
df = pd.read_csv(filename)

filename_labels = 'labels.csv'
df2 = pd.read_csv(filename_labels)

# Pandas returns a dataframe, (df). Convert it to numpy array
raw_data = df.values

cols = range(1, raw_data.shape[1])
X = raw_data[:, cols]
X = X.astype(float)

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

# Class index: string to numerical value
classLabels = df2.values[:,1] # -1 takes the last column
# finding unique class labels 
classNames = np.unique(classLabels)
# Dict: class name --> number
classDict = dict(zip(classNames,range(len(classNames))))

# Class index vector y_class (cancer types):
y_class = np.array([classDict[cl] for cl in classLabels])
# Regression vector y_reg (expression of a gene):
gene_reg = 15897
y_reg = X[:,gene_reg]

# Remove the gene used for regression from the data X and from attribute list
X = np.delete(X, gene_reg, axis=1)
attributeNames = np.delete(attributeNames, gene_reg)


# We can determine the number of data objects and number of attributes using 
# the shape of X
N, M = X.shape

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)

print('Data extracted')

#%% PCA

# Centering (Subtract mean value from data)
Y = X - np.ones((N,1))*X.mean(0)

# Standardize
#Y = Y*(1/np.std(Y,0))     # didn't work because some std's are zero
for s in range(N):
    Y_std = np.std(Y[:,s])
    if Y_std != 0:
        Y[:,s] = Y[:,s]*(1/Y_std)
    else:
        # removing the genes with StD=0 as they don't bring any information
        Y = np.delete(Y, s, axis=1)
        attributeNames = np.delete(attributeNames, s)
        


# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)

# rho -- explained variance of each principal component (vector)
rho = (S*S) / (S*S).sum() 
rho_cumsum = np.cumsum(rho)
threshold = 0.9

# find rho_sum that is higher than threshold
for i, rho_sum in enumerate(rho_cumsum):
    if rho_sum >= threshold:
        PC_n = i
        break


V = Vh.T    

# Project the centered data onto principal component space
Z = Y @ V

print('PCA finished')
