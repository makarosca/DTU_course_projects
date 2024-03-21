# RNA expression project
Training machine learning models on RNA expression dataset with the scope of identifying the best performing model.

The dataset used in this project is an RNA-Seq dataset samples from several different kinds of cancer: BRCA, KIRC, COAD, LUAD and PRAD. It has 801 samples, each having values for 20531 genes. (801 rows, 20531 columns)

This project consists of comparing two pairs of models:
1) Two classfication models for determination of cancer class based on the gene expression values: the multinomial logistic regression vs simple neural network.
2) Two regression models for predicting of a gene expression value based on the other gene expression values: the linear regression vs simple neural network.

For the more detailed description of the project, please read the project report.

The data can be downloaded here:
https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq
