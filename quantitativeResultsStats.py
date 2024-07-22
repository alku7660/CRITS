from data_constructor import Dataset
from model_constructor import Model
from interpreter_constructor import Interpreter
from address import results_scores,results_figures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import sparsity_score,pert_eval
import ipdb
import time 
import seaborn as sns

seed_int = 54321 # 54321

TRAIN_FRACTION = 0.8
SUBSET = 'test'
DATASETS = ['GunPointMaleVersusFemale','SharePriceIncrease','Strawberry','blink_EEG','SelfRegulationSCP1','Heartbeat']
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
MODELS_LABELS  = {'relu-mlp':'MLP LLM','conv-relu-mlp':'LICOR','CAM-conv':'CAM'} 
MODEL = 'conv-relu-mlp'
EXPLAINERS =["LICOR","SmoothGRAD","DeepSHAP","GradientSHAP"]

EXPLAINERS_LABELS = {"LICOR":"CRITS",
                    "SmoothGRAD":"SmoothGRAD",
                    "DeepSHAP":"DeepSHAP",
                    "GradientSHAP":"GradientSHAP"}

SCORES = [] 
SPARSITY = [] 
#N_TEST = len(data.test)
N_TEST = 50
TRIALS = 5
##########################################################################################
# PERTURBATION-BASED AND SPARSITY METRICS
##########################################################################################

def run_experiments(PATH_SCORES=False, PATH_SPARSITY=False):

    if not PATH_SCORES:
        
        for data_str in DATASETS:
            for trial in range(TRIALS):
                print(f"Trial {trial}/{TRIALS}")
                print(f"Loading data for dataset {data_str}")
                data = Dataset(data_str, TRAIN_FRACTION, seed_int)
                print(f"Loading model {MODEL} for dataset {data_str}")
                model_obj = Model(seed_int, data, MODEL)
                idx_list = range(N_TEST)
                #idx_list = range(5)
                _,test_auc = model_obj.train_model()
                interpreter = Interpreter(data, model_obj, idx_list, subset=SUBSET, use_bias=False)
                for explainer in EXPLAINERS:
                    print(f"interpreting dataset {data_str} using {explainer} explainer")
                    x_dict, x_label_dict, y_pred, weights_dict, intercept_dict, z_terms,w_range,z_range =interpreter.interpret_instances(explainer)
                    labels =np.array(list(x_label_dict.values()))
                    r = np.array(list(z_terms.values()))
                    x = np.array(list(x_dict.values()))
                    y = np.array(list(x_label_dict.values()))
                    x_pert,scores = pert_eval(x,y,r, model_obj,90)
                    SPARSITY.append([data_str,MODEL,explainer,sparsity_score(z_terms)])
                    for k,v in scores.items():
                        SCORES.append([data_str,MODEL,explainer,k,v])

        SCORES_DF = pd.DataFrame(data=SCORES,columns=["dataset","model","explainer","score","value"])
        SCORES_DF.to_csv(results_scores+f"{time.strftime('%Y_%m_%d_%H')}_perturbationSCORES_CNN_stats.csv",index=False)

        SPARSITY_DF = pd.DataFrame(data= SPARSITY, columns = ["dataset","model","explainer","value"])
        SPARSITY_DF.to_csv(results_scores+f"{time.strftime('%Y_%m_%d_%H')}_sparsity_CNN_stats.csv",index=False)

    else:

        SCORES_DF = pd.read_csv(results_scores+PATH_SCORES)
        SPARSITY_DF = pd.read_csv(results_scores+PATH_SPARSITY)

    return SCORES_DF, SPARSITY_DF

def print_latex_results(SCORES_DF, SPARSITY_DF):

    # REPLACING MODEL NAMES BY ITS LABELS 
    for expl in EXPLAINERS:
        SCORES_DF["explainer"].replace(to_replace=expl,value=EXPLAINERS_LABELS[expl],inplace=True)

    # PERTURBATION-BASED SCORES TABLE
    SCORES_TABLE_MEAN = pd.pivot_table(SCORES_DF,values='value',index='explainer',columns = ['dataset','score'])
    SCORES_TABLE_STD = pd.pivot_table(SCORES_DF,values='value',index='explainer',columns = ['dataset','score'],aggfunc='std')
    # # SCORES TABLE
    # SCORES_TABLE.to_latex(multicolumn=True)
    SCORES_TABLE_MEAN = SCORES_TABLE_MEAN[['blink_EEG','FingerMovements','SelfRegulationSCP1','SelfRegulationSCP2','Heartbeat']].T
    SCORES_TABLE_MEAN = SCORES_TABLE_MEAN[["SmoothGRAD","DeepSHAP","GradientSHAP","CRITS"]]

    SCORES_TABLE_STD = SCORES_TABLE_STD[['blink_EEG','FingerMovements','SelfRegulationSCP1','SelfRegulationSCP2','Heartbeat']].T
    SCORES_TABLE_STD = SCORES_TABLE_STD[["SmoothGRAD","DeepSHAP","GradientSHAP","CRITS"]]


    # print(SCORES_TABLE)
    # # SPARSITY TABLE
    SPARSITY_TABLE = pd.pivot_table(SPARSITY_DF,values='value',index='explainer',columns = 'dataset')
    SPARSITY_TABLE = SPARSITY_TABLE[['blink_EEG','FingerMovements','SelfRegulationSCP1','SelfRegulationSCP2','Heartbeat']].T
    SPARSITY_TABLE = SPARSITY_TABLE[["SmoothGRAD","DeepSHAP","GradientSHAP","LICOR"]]

    SPARSITY_TABLE_STD = pd.pivot_table(SPARSITY_DF,values='value',index='explainer',columns = 'dataset',aggfunc='std')
    SPARSITY_TABLE_STD = SPARSITY_TABLE_STD[['blink_EEG','FingerMovements','SelfRegulationSCP1','SelfRegulationSCP2','Heartbeat']].T
    SPARSITY_TABLE_STD = SPARSITY_TABLE_STD[["SmoothGRAD","DeepSHAP","GradientSHAP","LICOR"]]


def plot_results(SCORES_DF, SPARSITY_DF, name):

    # Boxplot figure
    plt.ion()
    sns.set_theme(style="ticks", palette="pastel")
    plt.close("Saliency metrics")
    plt.figure("Saliency metrics",figsize=(11,8))
    
    if 'SCORES' in name:
        RESULTS_DF = SCORES_DF
        ylabel_name = 'Alignment (A)'
    elif 'SPARSITY' in name:
        RESULTS_DF = SPARSITY_DF
        ylabel_name = 'Sparsity (S)'
    
    RESULTS_DF.loc[RESULTS_DF['explainer'] == 'LICOR','explainer'] = 'CRITS'

    for ii, dataset in enumerate(DATASETS):
        ax = plt.subplot(3,2,ii+1)
        data = RESULTS_DF.loc[RESULTS_DF["dataset"]==dataset]
        if 'SCORES' in name:
            ax = sns.boxplot(x="score", y="value", hue="explainer", palette=["m","g","r","b"], data=data, fliersize=0)
        elif 'SPARSITY' in name:
            ax = sns.barplot(x="explainer", y="value", data=data, errorbar=None)
        ax.set_xlabel('')
        ax.set_title(DATASET_LABELS[dataset],fontdict={"fontsize":13})
        # ax.legend(bbox_to_anchor=(1., 1.05))
        ax.set_ylabel(ylabel_name)
        if ii != 1 and 'SCORES' in name:
            ax.get_legend().remove()
        plt.subplots_adjust(left=0.09,bottom=0.06,right=0.96,top=0.97,wspace=0.175,hspace=0.45)
        sns.despine(offset=10, trim=False)
    
    if 'SCORES' in name:
        plt.savefig(fname=results_figures+PATH_SCORES[:-4]+".pdf",format='pdf')
    elif 'SPARSITY' in name:
        plt.savefig(fname=results_figures+PATH_SPARSITY[:-4]+".pdf",format='pdf')

PATH_SCORES = "2024_06_28_13_perturbationSCORES_CNN_stats.csv"
PATH_SPARSITY = "2024_06_28_13_sparsity_CNN_stats.csv"
# PATH_SCORES = False
# PATH_SPARSITY = False

SCORES_DF, SPARSITY_DF = run_experiments(PATH_SCORES, PATH_SPARSITY)
name = 'SPARSITY'
plot_results(SCORES_DF, SPARSITY_DF, name)


