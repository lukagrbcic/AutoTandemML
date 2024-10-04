import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from indago import PSO
import random

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from goal_functions import goal_function

class modelHCSampler:
    
    def __init__(self, model, sample_size, 
                 lb, ub, algorithm, function='uncertainty', x_sampled=[], beta=1):
        
        self.model = model
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        self.algorithm = algorithm
        self.function = function
        self.x_sampled = x_sampled
        self.beta = beta
        
    def train_calssifier(self):
        
        sampled_points = self.x_sampled
        # new_points = qmc.scale(qmc.LatinHypercube(d=len(self.lb)).random(n=len(sampled_points)), self.lb, self.ub)
        new_points = np.random.uniform(self.lb, self.ub, size=(len(sampled_points), len(self.lb)))
    
        ones = np.ones(len(sampled_points))
        zeros = np.zeros(len(sampled_points))
        data_ones = np.hstack((sampled_points, ones.reshape(-1,1)))
        data_zeros = np.hstack((new_points, zeros.reshape(-1,1)))
        
        data = np.vstack((data_ones, data_zeros))
        
        
        # classifier = RandomForestClassifier().fit(data[:, :-1], data[:,-1])
        classifier = xgb.XGBClassifier().fit(data[:, :-1], data[:,-1])
        
        return classifier

    def get_samples(self):

        if np.random.uniform() < self.beta:
            classifier = self.train_calssifier()
        else: classifier = None
        
        X = []
        f = []
        samples = qmc.scale(qmc.LatinHypercube(d=len(self.lb)).random(n=self.sample_size), self.lb, self.ub)
    
        for s in samples:
    
            def get_values(x):
                
                if classifier != None:
                    prediction = classifier.predict_proba([x])
                    if prediction[0,0] < 0.5:
                        value = 0
                        #p = 0
                    else:
                        # p = []
                        # if len(self.x_sampled) > 0:
                        #     for convbest in self.x_sampled:
                        #         val = np.linalg.norm(convbest - x)
                        #         p.append((1/val)**2)  
                        
                        preds = np.concatenate(np.array([model.predict([x]) for model in self.model.estimators_]))
    
                        value = goal_function(method=self.function).calculate(preds)
           
                return value #+ value*np.sum(p)

            # xs = []
            # fs = []
            
            # for i in range(80):
            #     dim = len(self.lb)
            #     min_f = np.inf
            #     min_x = None
                
                
            #     x0 = np.random.uniform(self.lb, self.ub, (1, dim))[0]
            #     bounds = np.array([[self.lb[i], self.ub[i]] for i in range(len(self.lb))])
            #     res = minimize(get_values, x0=x0, bounds=bounds, method='L-BFGS-B')
            #     if res.fun < min_f:
            #         min_f = res.fun
            #         min_x = res.x
                
            #     xs.append(min_x)
            #     fs.append(min_f)
                
            # min_f_index = np.argmin(fs)
            # min_f = fs[min_f_index]
            # min_x = xs[min_f_index]
                                
            optimizer = PSO()
            optimizer.evaluation_function = get_values 
            optimizer.lb = self.lb
            optimizer.ub = self.ub
            optimizer.max_evaluations = 30
            result = optimizer.optimize()
            min_x = result.X 
            min_f = result.f
                     
            X.append(min_x)
            f.append(min_f)
                
                
        # print (f)
        # print (X)
        
        # sort = np.argsort(f)[:self.sample_size]
        # X = X[sort]
        
        
        # X_f = np.hstack((X, f.reshape(-1,1)))
        # # cluster = KMeans(n_clusters=self.sample_size, n_init='auto').fit(X_f)
        # cluster = KMedoids(n_clusters=self.sample_size).fit(X_f)

    
        # X = cluster.cluster_centers_[:, :-1]
        
        X = np.array(X)
        f = np.array(f)
        
        return X
            
            
        
        