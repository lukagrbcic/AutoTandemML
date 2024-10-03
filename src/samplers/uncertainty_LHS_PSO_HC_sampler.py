import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from indago import PSO, DE, RS
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import xgboost as xgb

# np.random.seed(random.randint(1,1209423))




class uncertaintyHCSampler:
    
    def __init__(self, model, sample_size, lb, ub, algorithm, X_sampled, var=0.99, c=1):
        
        self.model = model
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        self.algorithm = algorithm
        self.X_sampled = X_sampled
        self.c = c
        self.var = var
        
    
    def generate_unique_points(self, initial_points, lower_bound, upper_bound, num_points):
        new_points = []
        initial_points_set = set(map(tuple, initial_points))
    
        while len(new_points) < num_points:
            point = np.random.uniform(lower_bound, upper_bound)
            point_tuple = tuple(point)
            
            if point_tuple not in initial_points_set:
                new_points.append(point)
                initial_points_set.add(point_tuple)  # To avoid duplicates in new_points
    
        return np.array(new_points)

    
    
    def train_calssifier(self):
        
        sampled_points = self.X_sampled
        # new_points = qmc.scale(qmc.LatinHypercube(d=len(self.lb)).random(n=len(sampled_points)), self.lb, self.ub)
        
                
        new_points = np.random.uniform(self.lb, self.ub, size=(len(sampled_points), len(self.lb)))
        
        # new_points = self.generate_unique_points(sampled_points, self.lb, self.ub, len(sampled_points))
        
        ones = np.ones(len(sampled_points))
        zeros = np.zeros(len(sampled_points))
               

        data_ones = np.hstack((sampled_points, ones.reshape(-1,1)))
        data_zeros = np.hstack((new_points, zeros.reshape(-1,1)))
        
        data = np.vstack((data_ones, data_zeros))
        
        
        # classifier = RandomForestClassifier().fit(data[:, :-1], data[:,-1])
        classifier = xgb.XGBClassifier().fit(data[:, :-1], data[:,-1])

        
        return classifier

        
    def get_unc_samples(self):
        
        if np.random.uniform() < self.var:
            classifier = self.train_calssifier()
        else: classifier = None
        
        # print (classifier)
        
        X = []
        f = []
        samples = qmc.scale(qmc.LatinHypercube(d=len(self.lb)).random(n=self.sample_size), self.lb, self.ub)

        for s in samples:
        # for i in range(self.sample_size):
            def uncertainty(x):
                
                # p = []
                # if len(X) > 0:
                    
                #     for convbest in X:
                #         val = np.linalg.norm(convbest - x)
                #         p.append(1/val)    
       
                

         
                         
                if self.algorithm == 'rf':
                    tree_preds = np.array([tree.predict(x.reshape(1,-1)) for tree in self.model.estimators_])
                    std = np.sum(np.std(tree_preds, axis=0)[0])
                    
                
                if classifier != None:
                    prediction = classifier.predict_proba(x.reshape(1,-1))
                    # print (prediction)
                    # print (prediction[0,0])
                    if prediction[0,0] < 0.5:
                        std = 0
                        p = 0
                            
                return -std #+ 0.1*np.sum(p)

            
            # dim = len(self.lb)
            # min_f = 1
            # min_x = None
            
            # x0 = np.random.uniform(self.lb, self.ub, (1, dim))[0]
            # bounds = np.array([[self.lb[i], self.ub[i]] for i in range(len(self.lb))])
            
            # res = minimize(uncertainty, x0=x0, bounds=bounds, method='Nelder-Mead')
            # if res.fun < min_f:
            #     min_f = res.fun
            #     min_x = res.x

            # x0 = np.random.normal(loc=s, scale=0.2*s, size=(len(s), np.shape(samples)[1]))
            # x0 = np.random.normal(loc=s, scale=0.2*s, size=(3, np.shape(samples)[1]))

            # x0 = np.clip(x0, self.lb, self.ub)
                           
            optim = PSO()
            # optim.params['swarm_size'] = len(s)
            # optim.params['cognitive_rate'] = 2
            # optim.params['social_rate'] = 1
            optim.X0 = s
            optim.max_evaluations = 350#3*len(s)
            optim.lb = self.lb
            optim.ub = self.ub
            optim.evaluation_function = uncertainty 
            run = optim.optimize()
            min_x = run.X
            min_f = run.f    
                
            X.append(min_x)
            f.append(min_f)
            
        f = np.array(f)
        X = np.array(X)
        
        # print (f)
        # print (X)
        
        # sort = np.argsort(f)[:self.sample_size]
        # X = X[sort]
        
        
        # X_f = np.hstack((X, f.reshape(-1,1)))
        # # cluster = KMeans(n_clusters=self.sample_size, n_init='auto').fit(X_f)
        # cluster = KMedoids(n_clusters=self.sample_size).fit(X_f)

    
        # X = cluster.cluster_centers_[:, :-1]
        
        
        
                       
        return X
            
            
        
        