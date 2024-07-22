# Testing sensitivity scores (accuracy and explanation)

#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_constructor import Dataset
from model_constructor import Model
from interpreter_constructor import Interpreter
from utils import explanation_sensitivity_score
from address import results_plots, results_obj
import pickle
import seaborn as sns

#Parameters
seed_int = 12345
EXPLAINERS_LABELS = {"LICOR":"CRITS",
                    "SmoothGRAD":"SmoothGRAD",
                    "DeepSHAP":"DeepSHAP",
                    "GradientSHAP":"GradientSHAP"}
DATASET_LABELS = {"Sine":"Sine",
                  "SineRandom":"SineRandom",
                  "SineRandomPhase":"SineRandomPhase",
                  'GunPointMaleVersusFemale':'GunPoint',
                  'SharePriceIncrease':'SharePriceIncrease',
                  'Strawberry':'Strawberry',
                  "blink_EEG":"Blink",
                  "FingerMovements":"FingerMovements",
                  "SelfRegulationSCP1":"SCP1",
                  "SelfRegulationSCP2":"SCP2",
                  "Heartbeat":"Heartbeat"
                  }

def run_sensitivity_experiments(data_names, explainers, std_dev_list, train_fraction, model_type):
    for data_name in data_names:
        data = Dataset(dataset_name=data_name, train_fraction=train_fraction, seed=seed_int)
        model = Model(seed=seed_int, data=data, model_type=model_type)
        _,test_auc = model.train_model()
        idx_list = list(range(30)) #len(data.test)
        interpreter = Interpreter(data=data, model=model, idx_list=idx_list)
        for explainer in explainers:
            explanation_score_mean, explanation_score_std = {}, {}
            for std_dev in std_dev_list:
                sensitivity_scores = list(explanation_sensitivity_score(interpreter=interpreter, explainer=explainer, std_dev=std_dev).values())
                explanation_score_mean[std_dev] = np.mean(sensitivity_scores)
                explanation_score_std[std_dev] = np.std(sensitivity_scores, ddof=1)
                print(f'----------------------------------------------------------------------')
                print(f'Standard Deviation done: Dataset {data_name} and Std. Dev.: {std_dev}')
                print(f'----------------------------------------------------------------------')
            with open(f'{results_obj}{data_name}_{model_type}_{explainer}_mean', 'wb') as handle:
                pickle.dump(explanation_score_mean, handle)
            with open(f'{results_obj}{data_name}_{model_type}_{explainer}_std', 'wb') as handle:
                pickle.dump(explanation_score_std, handle)

def plot_sensitivity_experiment_results(data_names, explainers, model_type):
    plt.ion()
    sns.set_theme(style="ticks", palette="pastel")
    plt.close("Saliency metrics")
    plt.figure("Saliency metrics",figsize=(11,8))
    palette=["m","g","r","b"]
    for idx, data_name in enumerate(data_names):
        ax = plt.subplot(3,2,idx+1)
        for idx_ex, explainer in enumerate(explainers):
            with open(f'{results_obj}{data_name}_{model_type}_{explainer}_mean', 'rb') as handle:
                explanation_score_mean = pickle.load(handle)
            with open(f'{results_obj}{data_name}_{model_type}_{explainer}_std', 'rb') as handle:
                explanation_score_std = pickle.load(handle)
            std_dev_list = list(explanation_score_mean.keys())
            ax.set_title(DATASET_LABELS[data_name],fontdict={"fontsize":13})
            ax.set_yscale('log')
            ax.set_xscale('log')
            y_mean = np.array(list(explanation_score_mean.values()))
            y_std = np.array(list(explanation_score_std.values()))
            y_low = y_mean - y_std/2
            y_low_n = []
            for i in y_low:
                if i > 0:
                    y_low_n.append(i)
                else:
                    y_low_n.append(0.000005)
            y_low = np.array(y_low_n)
            y_upp = y_mean + y_std/2
            ax.set_xlim(0.00005, 0.1)
            explainer_name = EXPLAINERS_LABELS[explainer]
            ax.plot(std_dev_list, y_mean, label=explainer_name, color=palette[idx_ex])
            # ax.fill_between(std_dev_list, y1=y_low, y2=y_upp, alpha=0.4, color=palette[idx_ex])
        ax.legend()
        if idx == len(data_names)-1 or idx == len(data_names)-2:
            ax.set_xlabel(f'Standard deviation [log]')
        ax.set_ylabel(f'Input-Sensitivity (IS) [log]')
        if idx != 1:
            ax.get_legend().remove()
    plt.subplots_adjust(left=0.09,bottom=0.06,right=0.96,top=0.97,wspace=0.175,hspace=0.45)
    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    plt.savefig(f'{results_plots}input-sensitivity.pdf')

data_names = ['GunPointMaleVersusFemale','SharePriceIncrease','Strawberry','blink_EEG','SelfRegulationSCP1','Heartbeat'] # 'GunPointMaleVersusFemale','SharePriceIncrease','Strawberry', 'blink_EEG', 'SelfRegulationSCP1', 'Heartbeat'
model_type = 'conv-relu-mlp' # 'relu-mlp'
train_fraction = 0.8
explainers = ['LICOR','SmoothGRAD','DeepSHAP','GradientSHAP'] #'Rectifier', 'SmoothGRAD', 'DeepSHAP', 'GradientSHAP'
std_dev_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1] # [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

# run_sensitivity_experiments(data_names=data_names, explainers=explainers, std_dev_list=std_dev_list, train_fraction=train_fraction, model_type=model_type)
plot_sensitivity_experiment_results(data_names=data_names, explainers=explainers, model_type=model_type)