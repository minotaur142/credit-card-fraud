# credit-card-fraud
An approach to the kaggle credit card fraud competition based on an ensemble of methods rare events (0.17% rows are fraud):

1. Bagged predictions from three penalized maximum likelihood estimation techniques (firth's logit, tuneable firth's logit, firth's logit with intercept correction).
2. L2 regularized logistic regression with synthetic minority oversampling technique (SMOTE)
3. Anomaly detection using stacked autencoders
4. Class weighted multi-layer perceptron

Achieves an AUC ROC score of 0.9884
