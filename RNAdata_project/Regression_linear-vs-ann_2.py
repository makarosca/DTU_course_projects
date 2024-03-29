# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:06:22 2022

@author: Asus
"""

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn import model_selection, linear_model as lm
from toolbox_02450 import rlr_validate, train_neural_net
import numpy as np
import torch

from ExtractData_PCA import Y, Z, y_reg as y, attributeNames as AtN, \
    gene_reg, PC_n

    
#%%

def ann_model_selection(X, y, K, h_values):
    

    N, M = X.shape
    C = 2
    
    
    # Parameters for neural network classifier
    #n_hidden_units = h      # number of hidden units
    n_replicates = 3        # number of networks trained in each k-fold
    max_iter = 5000
    
    # K-fold crossvalidation
    #K = 2                  # only three folds to speed up this example
    CV = model_selection.KFold(K, shuffle=True)
    
    # Setup figure for display of learning curves and error rates in fold
    summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
    # Make a list for storing assigned color of learning curve for up to K=10
   
    e_gen = np.empty((len(h_values),1))

    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
        
    errors = np.empty((K, len(h_values))) # make a list for storing generalizaition error in each loop
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))  
        
        for i, n_hidden_units in enumerate(h_values):
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            
            print('Training model of type:\n\n{}\n'.format(str(model())))
        
            # Extract training and test set for current CV fold, convert to tensors
            X_train = torch.Tensor(X[train_index,:])
            y_train = torch.Tensor(y[train_index])
            X_test = torch.Tensor(X[test_index,:])
            y_test = torch.Tensor(y[test_index])
            
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            
            print('\n\tBest loss: {}\n'.format(final_loss))
            
            # Determine estimated class labels for test set
            y_test_est = net(X_test).squeeze()
            
            # Determine errors and errors
            se = (y_test_est.float()-y_test.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
            errors[k, i] = mse # store error rate for current CV fold 
            # store mean error for the current fold instead:
            #errors.append(np.mean(mse))
            
        #e_gen[i] = round(np.sqrt(np.mean(errors)), 4)
    print(f'errors = {errors}')
    for i in range(len(h_values)):
        e_gen[i] = np.mean(errors[:, i])
    print(f'e_gen = {e_gen}')
    optimal_i = np.argmin(e_gen)
    print(f'h_values[optimal_i={optimal_i}] = {h_values[optimal_i]}')
    return h_values[optimal_i]


def ann_model_test(X_train, y_train, X_test, y_test, n_hidden_units):
    print(f'n_hidden_units in ann_model_test = {n_hidden_units}')
    N, M = X_train.shape
    n_replicates = 2        # number of networks trained in each k-fold
    max_iter = 5000
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    
    
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test).squeeze()
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean

    return y_test_est, se, mse


#%%

X = Z[:,:PC_n]
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+AtN
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,np.arange(-1,5,0.5))
h_values = [1, 5, 20]

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))
opt_lambda = []

E_test_ANN = np.empty((K,1))
h_opt = []




k=0
for train_index, test_index in CV.split(X,y):
    print(f'\n------- OUTER FOLD {k+1}/{K} -------')
    
    # extract training and test set for current CV fold
    X_train = np.copy(X[train_index])
    y_train = np.copy(y[train_index])
    X_test = np.copy(X[test_index])
    y_test = np.copy(y[test_index])
    internal_cross_validation = K   
    
    
    
    
    opt_val_err, opt_lambda_temp, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    opt_lambda.append(opt_lambda_temp)
    
    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda_temp * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    
    '''
    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    '''
    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda_temp)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))
    
    
    # ANN model ----------------
    
    h_opt_temp = ann_model_selection(np.copy(X[train_index]), np.copy(y[train_index]), K, h_values)
    h_opt.append(h_opt_temp)
    y_test_est_ANN, se_ANN, E_test_ANN[k] = ann_model_test(np.copy(X[train_index]), np.copy(y[train_index]), np.copy(X[test_index]), np.copy(y[test_index]), h_opt_temp)
    
    print(f'In fold {k+1}/{K}:')
    print(f'Linear regression. Test error = {Error_test_rlr[k]}; Optimal lambda = {opt_lambda[k]}')
    print(f'ANN regression. Test error = {E_test_ANN[k]}; Optimal h = {h_opt[k]}')      
    

    k+=1

show()
# Display results

for k in range(K):
    print(f'In fold {k+1}/{K}:')
    print(f'Linear regression. Test error = {Error_test_rlr[k]}; Optimal lambda = {opt_lambda[k]}')
    print(f'ANN regression. Test error = {E_test_ANN[k]}; Optimal h = {h_opt[k]}')      


    
# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of 
# the true/known value - these values should all be along a straight line "y=x", 
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value
plt.figure(figsize=(10,10))
y_est = X_test @ w_rlr[:,-1]; y_true = y_test
axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Alcohol content: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()

plt.show()

# y_test-X_test @ w_noreg[:,k]


#%%

print('ANN')
print(E_test_ANN)
print(h_opt)

print('Linear')
print(Error_test_rlr)
print(opt_lambda)

print('Baseline')
print(Error_test_nofeatures)


y_test_nofeat = y_test.mean()
y_test_rlr = X_test @ w_rlr[:,k]
y_test_ANN = y_test_est_ANN.detach().numpy()

zANN = np.abs(y_test_ANN - y[test_index] ) ** 2
zLR = np.abs(y_test_rlr - y[test_index] ) ** 2
zBase = np.abs(y[test_index] - y_test.mean()) ** 2

# ANN vs Linear
zA = zANN - zLR

alpha = 0.05
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval
pA = 2*st.t.cdf( -np.abs( np.mean(zA) )/st.sem(zA), df=len(zA)-1)  # p-value
print(pA)



# ANN vs Base
zB = zANN - zBase
CIB = st.t.interval(1-alpha, df=len(zB)-1, loc=np.mean(zB), scale=st.sem(zB))  # Confidence interval
pB = 2*st.t.cdf( -np.abs( np.mean(zB) )/st.sem(zB), df=len(zB)-1)  # p-value
print(pB)

# Linear vs Base

zC = zLR - zBase
CIC = st.t.interval(1-alpha, df=len(zC)-1, loc=np.mean(zC), scale=st.sem(zC))  # Confidence interval                  
pC = 2*st.t.cdf( -np.abs( np.mean(zC) )/st.sem(zC), df=len(zC)-1)  # p-value
print(pC)                  
                    

#%%
coef = w_rlr[:,-1]

biggest_coef = dict()


for i in range(len(w_rlr[:,-1])):
    if abs(w_rlr[:,-1][i]) > 10**(0):
        biggest_coef[i] = w_rlr[:,-1][i]

i = 1
for c in sorted(biggest_coef):
    print(f'{i}. PC_{c+1} = {w_rlr[:,-1][c]}')
    i += 1
