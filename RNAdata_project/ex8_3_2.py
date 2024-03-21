# exercise 8.3.2 Fit multinomial regression
from matplotlib.pyplot import figure, show, title
from scipy.io import loadmat
from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary
import numpy as np
import sklearn.linear_model as lm

from ExtractData_PCA import Z, y_class as y, attributeNames, \
    classNames, C, PC_n
    
from sklearn.model_selection import train_test_split



#%%

X = Z[:,:PC_n]
N, M = X.shape
C = len(classNames)


# Create crossvalidation partition for evaluation
# using stratification and 95 pct. split between training and test 
K = 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.90, stratify=y)

N, M = X.shape
C = len(classNames)
#%% Model fitting and prediction

# Multinomial logistic regression
logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, random_state=1)
logreg.fit(X_train,y_train)

# To display coefficients use print(logreg.coef_). For a 4 class problem with a 
# feature space, these weights will have shape (4, 2).

# Number of miss-classifications
print('Number of miss-classifications for Multinormal regression:\n\t {0} out of {1}'.format(np.sum(logreg.predict(X_test)!=y_test),len(y_test)))

predict = lambda x: np.argmax(logreg.predict_proba(x),1)
figure(2,figsize=(9,9))
visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
title('LogReg decision boundaries')

show()

print('Ran Exercise 8.3.2')