# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:30:34 2022

@author: Asus
"""

# exercise 1.5.1
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
import matplotlib.pyplot as plt
from scipy.linalg import svd

# Load the RNA csv file and its label file
filename = 'data.csv'
df = pd.read_csv(filename)

filename_labels = 'labels.csv'
df2 = pd.read_csv(filename_labels)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
raw_data = df.values

cols = range(1, raw_data.shape[1])
X = raw_data[:, cols]
X = X.astype(float)

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

# Before we can store the class index, we need to convert the strings that
# specify the class of a given object to a numerical value. We start by 
# extracting the strings for each sample from the raw data loaded from the csv:
classLabels = df2.values[:,1] # -1 takes the last column
# Then determine which classes are in the data by finding the set of 
# unique class labels 
classNames = np.unique(classLabels)
# We can assign each type of Iris class with a number by making a
# Python dictionary as so:
classDict = dict(zip(classNames,range(len(classNames))))

# With the dictionary, we can look up each data objects class label (the string)
# in the dictionary, and determine which numerical value that object is 
# assigned. This is the class index vector y:
y = np.array([classDict[cl] for cl in classLabels])
# In the above, we have used the concept of "list comprehension", which
# is a compact way of performing some operations on a list or array.
# You could read the line  "For each class label (cl) in the array of 
# class labels (classLabels), use the class label (cl) as the key and look up
# in the class dictionary (classDict). Store the result for each class label
# as an element in a list (because of the brackets []). Finally, convert the 
# list to a numpy array". 
# Try running this to get a feel for the operation: 
# list = [0,1,2]
# new_list = [element+10 for element in list]

# We can determine the number of data objects and number of attributes using 
# the shape of X
N, M = X.shape

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)


# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(0)

# boxplots for first 30 genes 
# (X - raw data, Y - centered)
plt.figure(figsize=(12,5))
plt.title('Gene expression distribution (Raw data)')
plt.boxplot(X[:,0:30])
#plt.xticks(np.arange(1,31), [f'Gene {i+1}' for i in range(30)])
plt.xlabel('Gene number')
plt.ylabel('Expression')
plt.show()



# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)




rho = (S*S) / (S*S).sum() 
threshold = 0.9

"""
# Plot variance explained by all principal components
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()
"""


# Variance explained by first PCs

threshold_cum = 0.5
threshold_single = 0.05
plt.figure()
plt.plot(range(1,11),rho[:10],'x-')
plt.plot(range(1,11),np.cumsum(rho[:10]),'o-')
plt.plot([1,10],[threshold_cum, threshold_cum],'k--')
plt.plot([1,10],[threshold_single, threshold_single],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()




# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = plt.figure()
title('Cancer data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.3)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()


""" #Same but standardized
# Standardize (make unit variance for all)
Y2 = np.copy(Y)
for s in range(N):
    Y_std = np.std(Y[:,s])
    if Y_std != 0:
        Y2[:,s] = Y[:,s]*(1/Y_std)
#Y2 = Y*(1/np.std(Y,0))     # didn't work cause some std's are zero

U2,S2,Vh2 = svd(Y2,full_matrices=False)
V2 = Vh2.T    

# Project the centered data onto principal component space
Z2 = Y2 @ V2

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = plt.figure()
title('Cancer data standardized: PCA')
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z2[class_mask,i], Z2[class_mask,j], 'o', alpha=.3)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))
show()

"""



"""
# Plot PCA of the data
f = plt.figure()
title('Z')
#Z = array(Z)
for i in range(10):
    for c in range(C):
        # select indices belonging to class c:
        class_mask = y==c
        Z1 = Z[class_mask,i]
        plot(np.ones(len(Z1))*i, Z1, 'o', alpha=.5)

# Output result to screen
show()


# Plot PCA of the data
f = plt.figure()
title('V[:,0]')
#Z = array(Z)
mask = 0.04
class_mask = V[:,0]>mask        # V[:,0]<-mask
class_mask2 = V[:,0]<-mask
plot(V[class_mask,0], 'o', alpha=.5)
plot(-V[class_mask2,0], 'o', alpha=.5)


# Output result to screen
show()
"""

"""
plt.figure()
plt.plot(V[:,0],'x-')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
#plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()
"""

# select principal component
PC_number = 3
# Find genes that account the most in the selected principal component
V0dict = dict(zip([i for i in range(len(V[:,PC_number]))], V[:,PC_number]))
# genes sorted by coefficients (absolute value) in PC
V_sorted_abs = sorted(V0dict, key=lambda key: abs(V0dict[key]), reverse=True)     


i = V_sorted_abs[2]
j = V_sorted_abs[1]
f = plt.figure()
title('Cancer data: Most significant genes')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(X[class_mask,i], X[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('Gene {0}'.format(i))
ylabel('Gene {0}'.format(j))

# Output result to screen
show()


"""#histogram
sample = 1

plt.figure(figsize=(12,4))
title('Normal distribution')
for i in range(10):
    plt.subplot(1,2,1)
    plt.plot(X[:,i],'.')
    plt.subplot(1,3,3)
    plt.hist(X[:,i], bins=30)
plt.show()
"""

""" histogram distribution of one gene, cancer types highlighted
gene_n = 3439
plt.figure()
title('Distribution of a particular gene')
for c in range(C):
    class_mask = y == c
    plt.hist(X[class_mask,gene_n], bins=30)
xlabel('Gene {0}'.format(gene_n))
legend(classNames)
plt.show()
"""

"""
gene_n = 3439
c = 2   # cancer type number
plt.figure()
title(f'Distribution of gene {gene_n} in {classNames[c]}')
class_mask = y == c
plt.boxplot(X[class_mask,gene_n])
xlabel('Gene {0}'.format(gene_n))
plt.show()
"""


"""
plt.figure()
plt.hist(V[0], bins=20)
plt.show()
"""


""" # boxplots for most significant genes in PC1
plt.figure(figsize=(25,10))
plt.boxplot(Y2[:,V0dict_sorted[0:10]])
plt.show()
"""

#[V0dict[i] for i in V_sorted_abs[:5]]
# Plot first 1 principal components
pcs = [0,1,2,3]
legendStrs = ['PC'+str(e+1) for e in pcs]
n_genes = 5     # number of (most important) genes to show in each component
plt.figure(figsize=(0.7*len(pcs)*n_genes,4))
#c = ['r','g','b']
bw = 0.2
# list for ticks on x-axis
ticks = []
# for every principal component
for n, PC_n in enumerate(pcs):
    # dict with keys=gene number; values=significance in principal component
    Vdict = dict(zip([j for j in range(len(V[:,PC_n]))], V[:,PC_n]))
    # gene numbers sorted by the absolute value of their significance in PC
    V_sorted_abs = sorted(Vdict, key=lambda key: abs(Vdict[key]), reverse=True)  
    # position on x-axis
    r = np.arange(n*n_genes+1, n*n_genes+1+n_genes)
    plt.bar(r, [Vdict[i] for i in V_sorted_abs[:n_genes]], width=bw)
    # list for xticks on x-axis. extend -- add components of new list
    ticks.extend(V_sorted_abs[:n_genes])
plt.xlabel('Gene numbers')
plt.ylabel('Component coefficients')
r = np.arange(1, n_genes*(len(pcs))+1)
plt.xticks(r, ticks)
plt.legend(legendStrs)
plt.grid()
plt.title('RNA Seq: PCA Component Coefficients')
plt.show()
