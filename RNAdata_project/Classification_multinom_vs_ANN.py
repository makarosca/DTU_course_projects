# exercise 8.1.2

import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import train_neural_net, mcnemar
from ExtractData_PCA import Z, y_class, attributeNames, \
    classNames, C, PC_n
import torch


#%%

def fit_log_reg(X, y, lambda_interval, K=5):    
    # find the optimal lambda for Multinomial Regression though internal CV
    
    CV = model_selection.KFold(K, shuffle=True)
    
    test_error_rate = np.zeros((K,len(lambda_interval)))
    
    for i, (train_index, test_index) in enumerate(CV.split(X,y)):
        #print(f'\n------- OUTER FOLD {k+1}/{K} -------')
        
        # extract training and test set for current CV fold
        X_train = np.copy(X[train_index])
        y_train = np.copy(y[train_index])
        X_test = np.copy(X[test_index])
        y_test = np.copy(y[test_index])
    
        # Standardize the training and set set based on training set mean and std
        mu = np.mean(X_train, 0)
        sigma = np.std(X_train, 0)
        
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma
        
        # Fit regularized logistic regression model to training data to predict 
        # the type of wine
        train_error_rate = np.zeros(len(lambda_interval))
        
        coefficient_norm = np.zeros(len(lambda_interval))
        for k in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k], 
                                     solver='lbfgs', multi_class='multinomial', 
                                     tol=1e-4, random_state=1)
            
            mdl.fit(X_train, y_train)
        
            y_train_est = mdl.predict(X_train)
            y_test_est = mdl.predict(X_test)
            
            train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
            test_error_rate[i,k] = np.sum(y_test_est != y_test) / len(y_test)
        
            w_est = mdl.coef_[0] 
            coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
        
        '''if i == 0:
            min_error = np.min(test_error_rate)
            plot_lambda(lambda_interval, train_error_rate, 
                            test_error_rate, min_error, coefficient_norm)'''

    
    opt_lambda = lambda_interval[np.argmin(np.mean(test_error_rate,axis=0))]

    return opt_lambda
   

'''def plot_lambda(lambda_interval, train_error_rate, 
                test_error_rate, min_error, coefficient_norm):
    
    print(lambda_interval, test_error_rate )
    plt.figure(figsize=(8,8))
    #plt.plot(np.log10(lambda_interval), train_error_rate*100)
    #plt.plot(np.log10(lambda_interval), test_error_rate*100)
    #plt.plot(np.log10(opt_lambda), min_error*100, 'o')
    plt.semilogx(lambda_interval, train_error_rate*100)
    plt.semilogx(lambda_interval, test_error_rate*100)
    plt.semilogx(opt_lambda, min_error*100, 'o')
    plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
    plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    plt.ylabel('Error rate (%)')
    plt.title('Classification error')
    plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
    #plt.ylim([0, 40])
    plt.grid()
    plt.show()    
    
    plt.figure(figsize=(8,8))
    plt.semilogx(lambda_interval, coefficient_norm,'k')
    plt.ylabel('L2 Norm')
    plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    plt.title('Parameter vector L2 norm')
    plt.grid()
    plt.show()'''


def ann_model_selection(X, y, K, h_values):   
    # find the optimal h value through internal CV
    
    N, M = X.shape

    # Since we're training a multiclass problem, we cannot use binary cross entropy,
    # but instead use the general cross entropy loss:
    loss_fn = torch.nn.CrossEntropyLoss()
    max_iter = 5000

    CV = model_selection.KFold(K,shuffle=True)

    errors = []     # make a list for storing generalizaition error in each loop
    errors = np.empty((K, len(h_values))) # make a list for storing generalizaition error in each loop
    e_gen = np.empty((len(h_values),1))


    for k, (train_index, test_index) in enumerate(CV.split(X,y)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        # Extract training and test set for current CV fold, 
        # and convert them to PyTorch tensors
        X_train = torch.Tensor(X[train_index,:] )
        y_train = torch.Tensor(y_class[train_index] )
        X_test = torch.Tensor(X[test_index,:] )
        y_test = y_class[test_index]
        
        print(f'Int. fold {k}: y_test ({len(y_test)}) = {y_test}')
        
        
        for i, n_hidden_units in enumerate(h_values):
            # Define the model
            model = lambda: torch.nn.Sequential(
                                        torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                                        torch.nn.ReLU(), # 1st transfer function
                                        # the nodes and their activation before the transfer 
                                        # function is often referred to as logits/logit output
                                        torch.nn.Linear(n_hidden_units, C), # C logits
                                        # To obtain normalised "probabilities" of each class
                                        # we use the softmax-funtion along the "class" dimension
                                        # (i.e. not the dimension describing observations)
                                        torch.nn.Softmax(dim=1) # final tranfer function, normalisation of logit output
                                        )
            
            # Train the network:
            net, final_loss, learning_curve = train_neural_net(model, loss_fn,
                                         X=torch.tensor(X_train, dtype=torch.float),
                                         y=torch.tensor(y_train, dtype=torch.long),
                                         n_replicates=3,
                                         max_iter=max_iter)
            # Determine probability of each class using trained network
            softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
            # Get the estimated class as the class with highest probability (argmax on softmax_logits)
            y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy() 
            # Determine errors
            e = (y_test_est != y_test)
            print(f'For h = {n_hidden_units}, y_test_est = {y_test_est}')
            print('Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(sum(e),len(e)))
            
            # store error rate for current CV fold  
            errors[k, i] = (sum(e)/len(y_test)) 
    for i in range(len(h_values)):
        e_gen[i] = np.mean(errors[:, i])
    print(f'e_gen = {e_gen}')
    
    # find index for optimal value for h
    optimal_i = np.argmin(e_gen)
    print(f'h_values[optimal_i={optimal_i}] = {h_values[optimal_i]}')
    return h_values[optimal_i]
    
    

def ann_model_test(X_train, y_train, X_test, y_test, n_hidden_units):
    
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = y_test
    
    model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                                torch.nn.ReLU(), # 1st transfer function
                                torch.nn.Linear(n_hidden_units, C), # C logits
                                torch.nn.Softmax(dim=1) # final tranfer function, normalisation of logit output
                                )
    loss_fn = torch.nn.CrossEntropyLoss()
    max_iter = 5000

    
    # Train the network:
    net, final_loss, learning_curve = train_neural_net(model, loss_fn,
                                 X=torch.tensor(X_train, dtype=torch.float),
                                 y=torch.tensor(y_train, dtype=torch.long),
                                 n_replicates=3,
                                 max_iter=max_iter)
    # Determine probability of each class using trained network
    softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
    # Get the estimated class as the class with highest probability (argmax on softmax_logits)
    y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy() 
    # Determine errors
    e = (y_test_est != y_test)
    print('Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(sum(e),len(e)))
    
    error_test = (sum(e)/len(y_test))
    
    return y_test_est, error_test


def biggest_class(y_train, y_test, C):
    # return a prediction equal to the biggest class.
    # Baseline model for multinomial prediction
    
    count = []   
    # find number of samples with each type of cancer
    for c in range(C):
        count.append(np.sum(y_train == c))
    
    # find the biggest class
    prediction = np.argmax(count)
    
    # assign this class to all predictions
    y_test_est = np.ones(len(y_test)) * prediction
    
    test_error = np.sum(y_test_est != y_test) / len(y_test)
    
    return y_test_est, test_error



    

    
#%%
#PC_n = 10
X = Z[:,:PC_n]
y = y_class

N, M = X.shape

# outer folds
K = 5
CV = model_selection.KFold(K, shuffle=True)

y_test_true = []

lambda_interval = np.logspace(-3, 3, 20)
opt_lambda = []
E_test_logreg = np.zeros((K,1))
y_test_logreg = []

h_values = [1, 20, 100]
h_opt = []
E_test_ANN = np.zeros((K,1))
y_test_ANN = []

E_test_base = np.zeros((K,1))
y_test_base = []


for k, (train_index, test_index) in enumerate(CV.split(X,y)):
    print(f'\n------- OUTER FOLD {k+1}/{K} -------')
    
    """# extract training and test set for current CV fold
    X_train = np.copy(X[train_index])
    y_train = np.copy(y[train_index])
    X_test = np.copy(X[test_index])"""
    y_test = np.copy(y[test_index])
    
    y_test_true.append(y_test)
    
    # --- logistic regression ---
    
    # find the optimal lambda for LogReg though internal CV
    opt_lambda_temp = fit_log_reg(np.copy(X[train_index]), 
                                  np.copy(y[train_index]), 
                                  lambda_interval, K)
    opt_lambda.append(opt_lambda_temp)
    
    mdl = LogisticRegression(penalty='l2', C=1/opt_lambda[k], solver='lbfgs', 
                             multi_class='multinomial', tol=1e-4, random_state=1)  
    mdl.fit(np.copy(X[train_index]), np.copy(y[train_index]))
    y_test_logreg_temp = mdl.predict(np.copy(X[test_index]))
    y_test_logreg.append(y_test_logreg_temp)
    E_test_logreg[k] = np.sum(y_test_logreg_temp != y_test) / len(y_test)
    
    
    # --- ANN for classification ---
    
    # train with selected h on X_train, predict X_test
    h_opt_temp = ann_model_selection(np.copy(X[train_index]),
                                     np.copy(y[train_index]), K, h_values)
    h_opt.append(h_opt_temp)
    y_test_ANN_temp, E_test_ANN[k] = ann_model_test(np.copy(X[train_index]),
                                                   np.copy(y[train_index]), 
                                                   np.copy(X[test_index]), 
                                                   np.copy(y[test_index]), 
                                                   h_opt[k])
    y_test_ANN.append(y_test_ANN_temp)
    
    # --- Baseline model ---
    
    y_test_base_temp, E_test_base[k] = biggest_class(np.copy(y[train_index]),
                                                     np.copy(y[test_index]),
                                                     C)
    y_test_base.append(y_test_base_temp)
    
    
    
y_test_logreg = np.concatenate(y_test_logreg)
y_test_ANN = np.concatenate(y_test_ANN)
y_test_base = np.concatenate(y_test_base)
y_test_true = np.concatenate(y_test_true)


# --- Statistical analysis by McNemar test ---

alpha = 0.05

# Multinomial vs ANN
[thetahatA, CIA, pA] = mcnemar(y_test_true, y_test_logreg, y_test_ANN, alpha=alpha)

# Multinomial vs Baseline
[thetahatB, CIB, pB] = mcnemar(y_test_true, y_test_logreg, y_test_base, alpha=alpha)

# ANN vs Baseline
[thetahatC, CIC, pC] = mcnemar(y_test_true, y_test_ANN, y_test_base, alpha=alpha)

    

#%%

# find biggest coefficients for multinomial regression

biggest_coef = dict()


for i in range(len(mdl.coef_)):
    for j in range(len(mdl.coef_[0])):
        if abs(mdl.coef_[i,j]) > 10**(-1.2):
            if biggest_coef.get(j) is None:
                biggest_coef[j] = []
            biggest_coef[j].append(i)

i = 1
for c in sorted(biggest_coef):
    for k in biggest_coef[c]:
        #print(f'{i}. PC_{c} in (class {classNames[k]}) = {mdl.coef_[k,c]}')
        print()
        i += 1
