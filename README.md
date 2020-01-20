# credit-card-fraud
An approach to the kaggle credit card fraud competition based on an ensemble of three methods for rare events:
1. Firth's Logistic Regression with Intercept Correction (FLIC)*
2. XGBoost with synthetic minority oversampling technique (SMOTE)
3. Anomaly detection using autencoders 

Missing values are handled using multiple imputation based on bagged random forests and linear regressors.

*FLIC accounts for bias due to imbalanced datasets by adding a 0.5log(|I|) to the log-likelihood function, which shrinks the coefficients in proportion to the inverse of the amount of information in the data, then reestimating the intercept so that the mean prediction matches the proportion of rare events in the data set.
