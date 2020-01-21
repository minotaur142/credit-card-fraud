import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import random
import statsmodels.api as sm
from utils import _add_constant,_hat_diag,_sigmoid_pred,_sigmoid_pred, _information_matrix, _predict, _predict_proba, _FLIC, _FLAC_aug


class PMLE():
    class Firth_Logit():
        def __init__(self,num_iters=10000, alpha=0.01,add_int=True,lmbda=0.5,FLAC=False, FLIC=False):
            
            '''PARAMETERS
               num_iters: number of iterations in gradient descent
               alpha: learning rate
               add_int: add intercept
               
               MODIFICATIONS FOR RARE EVENTS
               lmbda: tuneable parameter for target mean prediction value               
               FLAC: perform Firth Logistic regression with added covariate
               FLIC: perform Firth Logistic regression with Intercept Correction'''

            self.alpha = alpha
            self.num_iters = num_iters
            self.add_int = add_int
            self.lmbda=lmbda
            self.FLAC = FLAC
            self.FLIC=FLIC
        
        def firth_gd(self,X,y,weights):
            y_pred = _sigmoid_pred(X=X,weights=weights)
            H =_hat_diag(X,weights)
            I = _information_matrix(X,weights)
            U = np.matmul((y -y_pred + self.lmbda*H*(1 - 2*y_pred)),X)
            weights += np.matmul(np.linalg.inv(I),U)*self.alpha
            return weights
        
        def fit(self,X,y):
            #add intercept if necessary
            orig_X = X
            if self.add_int==True:
                X =_add_constant(X)
            self.X = X
            self.y = y

            #initialize weights
            weights=np.ones(X.shape[1])
            
            
            #Perform gradient descent
            for i in range(self.num_iters):
                weights = self.firth_gd(X,y,weights)
            
            if (self.FLAC==True)&(self.FLIC==True):
                X,y,aug_sample_weights=_FLAC_aug(X,y,weights)
                self.X = X
                self.y = y
                sklogit = LogisticRegression(solver='newton-cg',penalty='none',fit_intercept=False)
                sklogit.fit(X,y,sample_weight=aug_sample_weights)
                weights = sklogit.coef_[1:]
                eta = np.dot(orig_X,weights)
                target = y-eta
                b0_model = sm.OLS(target,np.ones(y.shape[0])).fit()
                b0 = b0_model.params[0]
                weights = np.insert(weights,0,b0)
            
            elif self.FLAC==True:
                X,y,aug_sample_weights=_FLAC_aug(X,y,weights)
                self.X = X
                self.y = y
                sklogit = LogisticRegression(solver='newton-cg',penalty='none',fit_intercept=False)
                sklogit.fit(X,y,sample_weight=aug_sample_weights)
                weights = sklogit.coef_
                
            
            elif self.FLIC==True:
                weights = weights[1:]
                eta = np.dot(orig_X,weights)
                target = y-eta
                b0_model = sm.OLS(target,np.ones(y.shape[0])).fit()
                b0 = b0_model.params[0]
                weights = np.insert(weights,0,b0)
            
            weights = pd.Series(weights.flatten(),index=self.X.columns)
            self.weights = weights
            
            I = _information_matrix(X,weights)
            hat_matrix_diag = _hat_diag(X,weights)
            Hessian = -I
            y_pred = _sigmoid_pred(X,weights)
            
            self.I = I
            self.hat_matrix_diag = hat_matrix_diag
            self.Hessian = Hessian
            
            
            self.log_likelihood = (y*np.log(y_pred)+(1-y)*np.log(1-y_pred)).sum()+0.5*np.log(np.linalg.det(I))
            
            
        def marginal_effects(self,values=None):
            '''PARAMETERS
               values: user-specified X values
               
               RETURNS
               marginal effects at mean X variable values
               mean of marginal effects for all rows
               marginal effects at user-specified values'''
                
            def at_specific_values(self,values):
                n_features = self.weights.shape[0]
                if values.shape[0]==n_features-1:
                    values = _add_constant(values)
                
                p = _sigmoid_pred(values,self.weights)
                effs = np.ones(n_features)
                for i in range(n_features):
                    weights_copy = self.weights.copy()
                    weights_copy[i]+=1
                    new_p =_sigmoid_pred(values,weights_copy)
                    effs[i] = new_p-p
                return effs
            
            #at mean column values
            column_means = self.X.mean()
            at_means = at_specific_values(weights=column_means)

            #find marginal effects for each row and take mean
            averaged_marg_effs = np.ones((self.X.shape[0],self.X.shape[1]))
            for i in range(self.X.shape[0]):
                row = self.X.iloc[i]
                p = _sigmoid_pred(row,self.weights)
                for j in range(self.weights.shape[0]):
                    weights_copy = self.weights.copy()
                    weights_copy[j]+=1
                    new_p =_sigmoid_pred(row,weights_copy)
                    eff = new_p-p
                    averaged_marg_effs[i,j] = eff
                ame = pd.DataFrame(averaged_marg_effs.mean(axis=0),index=self.X.columns, columns=['mean'])
                ame['at_means'] = at_means
            #user requested
            if (type(values)==numpy.ndarray) | (type(values)==pandas.core.series.Series):
                user_requested = at_specific_values(values)
                ame['requested_values'] = user_requsted
            return ame
        
        def predict(self,X):
            if self.FLAC==True:
                X = _FLAC_pred_aug(X)
            if X.shape[1]==self.X.shape[1]-1:
                X=_add_constant(X)
            return _predict(X,self.weights)
        
        def predict_proba(self,X):
            if self.FLAC==True:
                X = _FLAC_pred_aug(X)
            if X.shape[1]==self.X.shape[1]-1:
                X=_add_constant(X)
            return _predict_proba(X,self.weights)
        
    
    class logF11():
        def __init__(self,intercept=False):
            self.intercept=False
        
        def data_augementation(self,df,y_var_name):
            num_rows = 2*(df.shape[1]-1)
            y_ind = df.columns.get_loc(y_var_name)

            aug = pd.DataFrame(0,columns=df.columns,index=(range(num_rows)))

            #augment y variable
            aug.iloc[range(0,num_rows,2),y_ind]=1
            y = aug[y_var_name]

            #augment X variables
            X = aug.drop(y_var_name,axis=1)
            for ind, rows in enumerate(range(0,X.shape[0],2)):
                 X.iloc[rows:rows+2,ind]=1

            #bring it all together
            aug = pd.concat([y,X],axis=1)
            f_df = df.append(aug)

            #add offset
            f_df['real_data']=1
            f_df['real_data'][-aug.shape[0]:]=0
            f_df['real_data'].apply(lambda x: 0.5 if x == 0 else 1)

            #reseparate
            X = f_df.drop(y_var_name,axis=1)
            y = f_df[y_var_name]
            
            self.X = X
            self.y = y
    
            return X, y
        
        def fit(self,df,y_var_name):
            X, y = self.data_augementation(df,y_var_name)
            model = sm.Logit(y,X).fit()
            weights = model.params
            if self.intercept==True:
                weights = _FLIC(X,weights)
                X = _add_constant(X)
            self.X = X
            weights = pd.Series(weights,index=X.columns)
            self.weights = weights
        
        def predict(self,X):
            return _predict(X,self.weights)
        
        def predict_proba(self,X):
            return _predict_proba(X,self.weights)
            

