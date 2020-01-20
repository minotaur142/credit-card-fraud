# credit-card-fraud
An approach to the kaggle credit card fraud competition based on an ensemble of three methods for rare events:
1. Firth's Logistic Regression with Intercept Correction (penalized maximum likelihood estimation)
2. XGBoost with synthetic minority oversampling technique
3. Anomaly detection using autencoders 

Missing values are handled using multiple imputation based on bagged random forests and linear regressors.
