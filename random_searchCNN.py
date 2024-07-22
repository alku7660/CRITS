import warnings
warnings.filterwarnings("ignore")
import os

from address import results_grid_search
from data_constructor import Dataset
from tester import datasets, train_fraction, seed_int
from model_constructor import Model
from keras import callbacks
# from tensorflow.python import keras
from keras import callbacks
import pandas as pd
import numpy as np
import keras_tuner 

path_here = os.path.abspath('')
model_type = 'conv-relu-mlp' 
N_TRIALS = 150
EX_PER_TRIAL = 2
dataset_model = ""
results = pd.DataFrame(index=datasets, columns=['params','F1'])

# Hypermodel
def build_model(hp):
    global dataset_model
    # Dummy params to get the model
    params = {}
    params["explainers"] = ["CAM","SmoothGRAD","DeepSHAP","GradientSHAP"]
    # Dummy data to get the model
    data = Dataset(dataset_model, train_fraction, seed_int)
    
    n_timesteps = data.n_timesteps
    n_feats = data.n_feats
    size_time_kernel = hp.Int("size_time_kernel", min_value=3, max_value=16, step=1)
    n_kernels = hp.Int("n_kernels", min_value=16, max_value=256, step=16)
    neurons_l1 = hp.Int("neurons_l1", min_value=20, max_value=200, step=10)
    neurons_l2 = hp.Int("neurons_l2", min_value=20, max_value=200, step=10)
    neurons_l3 = hp.Int("neurons_l3", min_value=20, max_value=200, step=10)
    neurons_l4 = hp.Int("neurons_l4", min_value=20, max_value=200, step=10)
    neurons_l5 = hp.Int("neurons_l5", min_value=20, max_value=200, step=10)

    hidden_layers_sizes = (neurons_l1,neurons_l2,neurons_l3,neurons_l4,neurons_l5)
    activations = 'relu'
    # call existing model-building code with the hyperparameter values.
    model = Model(seed_int, data, model_type, params).create_conv_network(n_timesteps,
                                                                        n_feats,
                                                                        size_time_kernel,
                                                                        n_kernels,
                                                                        hidden_layers_sizes,
                                                                        activations)
    return model

for data_str in datasets: 
    data = Dataset(data_str, train_fraction, seed_int)
    dataset_model=data_str
    tuner = keras_tuner.RandomSearch(
                            hypermodel=build_model,
                            objective="val_loss",
                            max_trials=N_TRIALS,
                            executions_per_trial=EX_PER_TRIAL,
                            overwrite=True,
                            directory="./Results/tmpCNNSearch/",
                            project_name=f"{data_str}",
                        )
    tuner.search_space_summary()

    n_timesteps         = data.n_timesteps
    n_feats             = data.n_feats
    X_train             = data.train.reshape((-1,n_feats,n_timesteps))
    X_train             = np.transpose(X_train,(0,2,1)).reshape(-1,n_timesteps,1,n_feats)
    y_train             = data.train_target

    X_test              = data.test.reshape((-1,n_feats,n_timesteps))
    X_test              = np.transpose(X_test,(0,2,1)).reshape(-1,n_timesteps,1,n_feats)
    y_test              = data.test_target

    ES_cb = callbacks.EarlyStopping(monitor="val_loss",
                                    min_delta=0.001,
                                    patience=50,                                            
                                    baseline=None,
                                    restore_best_weights=True)
    tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test),callbacks=[ES_cb])
    bestParams = tuner.get_best_hyperparameters(1)[0]
    bestModel = tuner.get_best_models(1)[0]
    bestLoss,bestF1 = bestModel.evaluate(X_test,y_test)
    results.loc[data_str,'params'] = [{'solver': 'adam',
                                    'hidden_layers_sizes': (bestParams['neurons_l1'],bestParams['neurons_l2'],bestParams['neurons_l3'],bestParams['neurons_l4'],bestParams['neurons_l5']),
                                    "size_time_kernel":bestParams["size_time_kernel"],
                                    'n_kernels':bestParams["n_kernels"],
                                    'activation': 'relu'}]
    results.loc[data_str,'F1'] = bestF1
results.to_csv(f'{results_grid_search}grid_searchCNN_2.csv',mode='a')    


