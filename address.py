import os
import pickle
# import pickle5 as pickle

path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/Datasets/'
results_grid_search = str(path_here)+'/Results/Grid_search/'
results_obj = str(path_here)+'/Results/Obj/'
results_plots = str(path_here)+'/Results/Plots/'
results_scores = str(path_here)+'/Results/Scores/'
results_figures = str(path_here)+'/Results/Figures/'


def save_obj(address, obj, file_name):
    """
    Method to store an Evaluator object containing the evaluation results for all the instances of a given dataset
    """
    with open(address+file_name, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    
def load_obj(address, file_name):
    """
    Method to read an Evaluator object containing the evaluation results for all the instances of a given dataset
    """
    with open(address+file_name, 'rb') as input:
        obj = pickle.load(input)
    return obj

def select_models(address, file = None):
    """
    Method to select the models to test. If there exists a best model, attempt to get it, else use the list provided below
    """
    if file is None:
        models = ['lin','elm','rf','lin_ens','mlp-relu']
    else:
        with open(address+file, 'rb') as input:
            models = [pickle.load(input)[0]]
    return models