# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:40:39 2022

@author: Asus
"""
import matplotlib.pyplot as plt #import figure, show, title

from sklearn import model_selection
from toolbox_02450 import dbplotf, train_neural_net, draw_neural_net
import numpy as np
import torch

from ExtractData_PCA import Z, y_class, y_reg, attributeNames, gene_reg, \
    classNames, C, PC_n

#%% 

X = Z[:,:PC_n]

N, M = X.shape

# Define the model structure
n_hidden_layer_1 = 10 # number of hidden units in the signle hidden layer
n_hidden_layer_2 = 10
model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, n_hidden_layer_1), #M features to H hiden units
                            torch.nn.ReLU(), # 1st transfer function
                            # Output layer:
                            # H hidden units to C classes
                            # the nodes and their activation before the transfer 
                            # function is often referred to as logits/logit output
                            
                            #torch.nn.Linear(n_hidden_layer_1, n_hidden_layer_2), 
                            #torch.nn.ReLU(), 
                            
                            torch.nn.Linear(n_hidden_layer_1, C), # C logits
                            # To obtain normalised "probabilities" of each class
                            # we use the softmax-funtion along the "class" dimension
                            # (i.e. not the dimension describing observations)
                            torch.nn.Softmax(dim=1) # final tranfer function, normalisation of logit output
                            )
# Since we're training a multiclass problem, we cannot use binary cross entropy,
# but instead use the general cross entropy loss:
loss_fn = torch.nn.CrossEntropyLoss()

max_iter = 4000

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

# K-fold CrossValidation (4 folds here to speed up this example)
K = 4
CV = model_selection.KFold(K,shuffle=True)

#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_class, test_size=0.33, random_state=42)

errors = [] # make a list for storing generalizaition error in each loop

# number of misclassifications for every fold
misclass = []


for k, (train_index, test_index) in enumerate(CV.split(X,y_class)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, 
    # and convert them to PyTorch tensors
    X_train = torch.Tensor(X[train_index,:] )
    y_train = torch.Tensor(y_class[train_index] )
    X_test = torch.Tensor(X[test_index,:] )
    y_test = y_class[test_index]
    
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
    misclass.append(sum(e))
    
    error_rate = (sum(e)/len(y_test))
    errors.append(error_rate) # store error rate for current CV fold 
    
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    


# Display the error rate across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold');
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('Error rate');
summaries_axes[1].set_title('Test misclassification rates')



# Print the average classification error rate
print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors),4)))








#predict = lambda x:  (torch.max(net(torch.tensor(x, dtype=torch.float)), dim=1)[1]).data.numpy() 
#figure(1,figsize=(9,9))
#visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
#title('ANN decision boundaries')




# Display a diagram of the best network in last fold
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,3]]
draw_neural_net(weights, biases, tf)

plt.show()


