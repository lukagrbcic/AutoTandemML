import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import *
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, '../InverseBench/src/')
sys.path.insert(1, 'src')
sys.path.insert(2, 'src/samplers')

from benchmarks import *
from auto_tandem import AutoTNN
from ensemble_regressor import EnsembleRegressor
from inverse_validator import inverse_model_analysis


import warnings
warnings.filterwarnings("ignore")


# print ('airfoil')


bench = 'airfoils' #(xgb ensemble)
name = 'airfoil_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
# ensemble = [XGBRegressor(n_estimators=i[1], reg_lambda=i[0]) for i in [[0.1, 10], [0.5,50], [0.8, 75], [1,100], [10, 125]]]   

# ensemble_size = 10
# n_est = np.arange(10, 210, ensemble_size)
# reg_lambda = np.linspace(0.1, 10, ensemble_size)
# list_ = [[reg_lambda[i], n_est[i]] for i in range(ensemble_size)]

# ensemble = [XGBRegressor(n_estimators=i[1], reg_lambda=i[0]) for i in list_]             
          
# algorithm = ('xgb_ensemble', EnsembleRegressor(ensemble))

# algorithm = ('rf', RandomForestRegressor())


bench = 'friedman' #(deep ensembles)
name = 'friedman_multioutput_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
algorithm = ('rf', RandomForestRegressor())

# ensemble = []
# for i in range(20):
#     ensemble.append(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 200, 100), 
#                                                                   random_state=i)))
# algorithm = ('mlp_ensemble', EnsembleRegressor(ensemble)) 

# algorithm = ('rf', RandomForestRegressor())


# bench = 'inconel' #(random forests)
# name = 'inconel_benchmark'
# model = load_model(name).load()
# f = benchmark_functions(name, model)
# algorithm = ('rf', RandomForestRegressor())

lb, ub = f.get_bounds()

test_input = np.load(f'../InverseBench/test_data/{bench}_data/input_test_data.npy')[:1000]
test_output = np.load(f'../InverseBench/test_data/{bench}_data/output_test_data.npy')[:1000]
test_data = (test_input, test_output)

init_size=20
batch_size=10
max_samples=200

# sampler='model_uncertainty'
sampler='model_quantile'

r2_ = []
rmse_ = []
mape_ = []
nmax_ae_ = []
runs = 10
n = 20

for i in range(runs):
    print ('Run', i+1)

    run = AutoTNN(f, lb, ub, init_size, batch_size, max_samples, algorithm, test_data, 
                  sampler=sampler, combinations=n)
    run.get_inverse_DNN()
    r2, rmse, mape, nmax_ae = inverse_model_analysis(test_input, test_output, name, sampler).error_metrics()
    
    print ('R2:', r2)
    print ('RMSE:', rmse)
    print ('MAPE:', mape)
    print ('NMAX_AE:', nmax_ae)

    r2_.append(r2)
    rmse_.append(rmse)
    mape_.append(mape)
    nmax_ae_.append(nmax_ae)

print (sampler)
print ('R2:', np.mean(r2_), np.std(r2_))
print ('RMSE:', np.mean(rmse_), np.std(rmse_))
print ('MAPE:', np.mean(mape_), np.std(mape_))
print ('NMAX_AE:', np.mean(nmax_ae_), np.std(nmax_ae_))


# for i in range(runs):
#     print ('Run', i+1)

#     run = AutoTNN(f, lb, ub, init_size, batch_size, max_samples, algorithm, test_data, lf_samples=1000,
#                   sampler=sampler, combinations=n)
#     run.get_inverse_DNN()
#     r2, rmse, mape, nmax_ae = inverse_model_analysis(test_input, test_output, name, sampler).error_metrics()
    
#     r2_.append(r2)
#     rmse_.append(rmse)
#     mape_.append(mape)
#     nmax_ae_.append(nmax_ae)

# print (sampler)
# print ('R2:', np.mean(r2_), np.std(r2_))
# print ('RMSE:', np.mean(rmse_), np.std(rmse_))
# print ('MAPE:', np.mean(mape_), np.std(mape_))
# print ('NMAX_AE:', np.mean(nmax_ae_), np.std(nmax_ae_))


sampler='random'

r2_ = []
rmse_ = []
mape_ = []
nmax_ae_ = []

for i in range(runs):
    print ('Run', i+1)
    x_sampled_rand = np.random.uniform(lb, ub, size=(max_samples, len(lb)))
    y_sampled_rand = f.evaluate(x_sampled_rand)
    
    run = AutoTNN(f, lb, ub, init_size, batch_size, max_samples, algorithm, test_data, 
                  sampler=sampler, combinations=n, x_init=x_sampled_rand, y_init=y_sampled_rand)
    run = AutoTNN(f, lb, ub, init_size, batch_size, max_samples, algorithm, test_data, 
                  sampler=sampler, combinations=n)
    run.get_inverse_DNN()
    r2, rmse, mape, nmax_ae = inverse_model_analysis(test_input, test_output, name, sampler).error_metrics()
        
    print ('R2:', r2)
    print ('RMSE:', rmse)
    print ('MAPE:', mape)
    print ('NMAX_AE:', nmax_ae)

    r2_.append(r2)
    rmse_.append(rmse)
    mape_.append(mape)
    nmax_ae_.append(nmax_ae)

print (sampler)
print ('R2:', np.mean(r2_), np.std(r2_))
print ('RMSE:', np.mean(rmse_), np.std(rmse_))
print ('MAPE:', np.mean(mape_), np.std(mape_))
print ('NMAX_AE:', np.mean(nmax_ae_), np.std(nmax_ae_))


"""

print ('friedman')


bench = 'friedman' #(deep ensembles)
name = 'friedman_multioutput_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
ensemble = []
for i in range(20):
    ensemble.append(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 200, 100), 
                                                                  random_state=i)))
algorithm = ('mlp_ensemble', EnsembleRegressor(ensemble)) 



lb, ub = f.get_bounds()

test_input = np.load(f'../InverseBench/test_data/{bench}_data/input_test_data.npy')
test_output = np.load(f'../InverseBench/test_data/{bench}_data/output_test_data.npy')
test_data = (test_input, test_output)

init_size=20
batch_size=10
max_samples=300

sampler='model_uncertainty'

r2_ = []
rmse_ = []
mape_ = []
nmax_ae_ = []
runs = 10
n = 100

for i in range(runs):
    run = AutoTNN(f, lb, ub, init_size, batch_size, max_samples, algorithm, test_data, 
                  sampler=sampler, combinations=n)
    run.get_inverse_DNN()
    r2, rmse, mape, nmax_ae = inverse_model_analysis(test_input, test_output, name, sampler).error_metrics()
    
    r2_.append(r2)
    rmse_.append(rmse)
    mape_.append(mape)
    nmax_ae_.append(nmax_ae)

print (sampler)
print ('R2:', np.mean(r2_), np.std(r2_))
print ('RMSE:', np.mean(rmse_), np.std(rmse_))
print ('MAPE:', np.mean(mape_), np.std(mape_))
print ('NMAX_AE:', np.mean(nmax_ae_), np.std(nmax_ae_))

sampler='random'

r2_ = []
rmse_ = []
mape_ = []
nmax_ae_ = []

for i in range(runs):
    print ('Run', i+1)
    
    x_sampled_rand = np.random.uniform(lb, ub, size=(max_samples, len(lb)))
    y_sampled_rand = f.evaluate(x_sampled_rand)
    
    run = AutoTNN(f, lb, ub, init_size, batch_size, max_samples, algorithm, test_data, 
                  sampler=sampler, combinations=n, x_init=x_sampled_rand, y_init=y_sampled_rand)
    run.get_inverse_DNN()
    r2, rmse, mape, nmax_ae = inverse_model_analysis(test_input, test_output, name, sampler).error_metrics()
    
    r2_.append(r2)
    rmse_.append(rmse)
    mape_.append(mape)
    nmax_ae_.append(nmax_ae)

print (sampler)
print ('R2:', np.mean(r2_), np.std(r2_))
print ('RMSE:', np.mean(rmse_), np.std(rmse_))
print ('MAPE:', np.mean(mape_), np.std(mape_))
print ('NMAX_AE:', np.mean(nmax_ae_), np.std(nmax_ae_))



print ('Inconel')

# bench = 'inconel' #(random forests)
# name = 'inconel_benchmark'
# model = load_model(name).load()
# f = benchmark_functions(name, model)
# algorithm = ('rf', RandomForestRegressor())

lb, ub = f.get_bounds()

test_input = np.load(f'../InverseBench/test_data/{bench}_data/input_test_data.npy')
test_output = np.load(f'../InverseBench/test_data/{bench}_data/output_test_data.npy')
test_data = (test_input, test_output)

init_size=20
batch_size=10
max_samples=300

sampler='model_uncertainty'

r2_ = []
rmse_ = []
mape_ = []
nmax_ae_ = []
runs = 10
n = 100

for i in range(runs):
    run = AutoTNN(f, lb, ub, init_size, batch_size, max_samples, algorithm, test_data, 
                  sampler=sampler, combinations=n)
    run.get_inverse_DNN()
    r2, rmse, mape, nmax_ae = inverse_model_analysis(test_input, test_output, name, sampler).error_metrics()
    
    r2_.append(r2)
    rmse_.append(rmse)
    mape_.append(mape)
    nmax_ae_.append(nmax_ae)

print (sampler)
print ('R2:', np.mean(r2_), np.std(r2_))
print ('RMSE:', np.mean(rmse_), np.std(rmse_))
print ('MAPE:', np.mean(mape_), np.std(mape_))
print ('NMAX_AE:', np.mean(nmax_ae_), np.std(nmax_ae_))

sampler='random'

r2_ = []
rmse_ = []
mape_ = []
nmax_ae_ = []

for i in range(runs):
    print ('Run', i+1)
    
    x_sampled_rand = np.random.uniform(lb, ub, size=(max_samples, len(lb)))
    y_sampled_rand = f.evaluate(x_sampled_rand)
    
    run = AutoTNN(f, lb, ub, init_size, batch_size, max_samples, algorithm, test_data, 
                  sampler=sampler, combinations=n, x_init=x_sampled_rand, y_init=y_sampled_rand)
    run.get_inverse_DNN()
    r2, rmse, mape, nmax_ae = inverse_model_analysis(test_input, test_output, name, sampler).error_metrics()
    
    r2_.append(r2)
    rmse_.append(rmse)
    mape_.append(mape)
    nmax_ae_.append(nmax_ae)

print (sampler)
print ('R2:', np.mean(r2_), np.std(r2_))
print ('RMSE:', np.mean(rmse_), np.std(rmse_))
print ('MAPE:', np.mean(mape_), np.std(mape_))
print ('NMAX_AE:', np.mean(nmax_ae_), np.std(nmax_ae_))



"""


