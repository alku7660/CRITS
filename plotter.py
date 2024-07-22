import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from address import load_obj, results_plots, results_obj, dataset_dir
from tester import datasets, idx_list
import seaborn as sns
gray_cmap=LinearSegmentedColormap.from_list('gy',[(.3,.3,.3),(.8,.8,.8)], N=2)
colors = ['blue', 'red', 'brown', 'purple', 'green', 'mediumpurple', 'darkgreen', 'firebrick', 'black', 'gray']

def make_meshgrid(x1, x2, h=.02, x1_min=0, x1_max=0, x2_min=0, x2_max=0):
    """
    Credits to Zed Lee for this 2D plotting function
    """
    if x1_min==0 and x1_max==0 and x2_min==0 and x2_max==0:
        x1_min, x1_max = x1.min() - 0.1, x1.max() + 0.1
        x2_min, x2_max = x2.min() - 0.1, x2.max() + 0.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    return np.vstack((xx1.ravel(), xx2.ravel())).T

def scatter_2d(interpreter):
    fig_2d, ax_2d = plt.subplots(figsize=(3,3))
    train = interpreter.train
    global_model = interpreter.model
    x_mesh = make_meshgrid(train[:,0],train[:,1])
    y_mesh = global_model.classifier.predict(x_mesh)
    y_pert_global_pred = global_model.classifier.predict(train)
    ax_2d.scatter(x_mesh[:,0], x_mesh[:,1], s=12, c=y_mesh, cmap=gray_cmap)
    ax_2d.scatter(train[:,0], train[:,1], c=y_pert_global_pred, linewidths=0.4, edgecolors='blue')
    values_x1 = np.linspace(min(x_mesh[:,0]), max(x_mesh[:,0]), num=100)
    ax_2d.set_ylim(bottom=min(train[:,0])-0.05, top=max(train[:,0])+0.05)
    ax_2d.set_xlim(left=min(train[:,0])-0.05, right=max(train[:,0])+0.05)
    ax_2d.axes.xaxis.set_visible(False)
    ax_2d.axes.yaxis.set_visible(False)
    # ax_2d.set_aspect('equal', 'box')
    for x_idx in interpreter.x_dict.keys():
        ioi = interpreter.x_dict[x_idx]
        m = -interpreter.weights_dict[x_idx][0]/interpreter.weights_dict[x_idx][1]
        b = -interpreter.intercept_dict[x_idx]/interpreter.weights_dict[x_idx][1]
        x_close = (m*ioi[1]+ioi[0]-m*b)/(m**2+1)
        y_close = m*x_close + b
        ax_2d.plot(values_x1, m*values_x1 + b, color='green', linewidth=0.75)
        ax_2d.scatter(x_close, y_close, c='green', marker='o')
        ax_2d.plot(x_close, y_close, marker='${}$'.format(x_idx), markersize=4, markeredgecolor='black', markeredgewidth=0.3, label=x_idx, color='black')
        ax_2d.scatter(ioi[0], ioi[1], c='white', marker='o', linewidths=0.4, edgecolors='blue')
        ax_2d.scatter(ioi[0], ioi[1], s=16, c='black', marker='${}$'.format(x_idx), label=x_idx, linewidths=0.3)
    # labellines.labelLines(ax_2d.get_lines())
    fig_2d.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.01)
    fig_2d.savefig(f'{results_plots}{interpreter.name}_{len(interpreter.x_dict.keys())}_2d.pdf')

def plot_helix(helix_idx_list, data_str):
    helix_df = pd.read_csv(f'{dataset_dir}helix/{data_str}.csv', index_col=0)
    number_helix = int((len(helix_df.columns) - 1)/72) #120
    range_helix = np.array(range(72)) #120
    fig, ax = plt.subplots(nrows=len(helix_idx_list), ncols=1, figsize=(7,10))
    for helix in range(number_helix):
        for ax_idx in range(len(helix_idx_list)):
            idx = helix_idx_list[ax_idx]
            ax[ax_idx].axes.xaxis.set_visible(False)
            ax[ax_idx].axes.yaxis.set_visible(False)
            columns_helix = range_helix + helix*72 #120
            columns_helix = [str(i) for i in columns_helix]
            helix_array = helix_df.loc[idx, columns_helix]
            label = helix_df.loc[idx,'label']
            ax[ax_idx].plot(range_helix, helix_array, color=colors[helix])
            ax[ax_idx].set_title(f'Helix instance {idx}. Label: {label}')
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.25)
    fig.savefig(f'{results_plots}{data_str}_{len(helix_idx_list)}.pdf')

def plot_helix_interpretability(interpreter, number_instances):
    fig_weights, ax_weights = plt.subplots(nrows=number_instances, ncols=1, figsize=(8,11))
    fig_z, ax_z = plt.subplots(nrows=number_instances, ncols=1, figsize=(8,11))
    # list_idx_instances = []
    # for idx in range(number_instances):
    #     x_idx = int(np.random.choice(list(interpreter.x_dict.keys())))
    #     list_idx_instances.extend([x_idx])
    # instances_weights = []
    # for idx in list_idx_instances:
    #     flat_weights = interpreter.weights_dict[idx]
    #     instances_weights.extend(flat_weights)
    # instances_weights = np.array(instances_weights)
    # min_weight, max_weight = min(instances_weights), max(instances_weights)
    list_idx = list(interpreter.x_dict.keys())
    label_0_idx_list = [i for i in list_idx if i%2 == 0]
    label_1_idx_list = [i for i in list_idx if i%2 == 1]
    for ax_idx in range(number_instances):
        ax_weights[ax_idx].axes.xaxis.set_visible(False)
        ax_z[ax_idx].axes.xaxis.set_visible(False)
        # x_idx = list_idx_instances[ax_idx]
        if ax_idx%2 == 0:
            x_idx = int(np.random.choice(label_0_idx_list))
        else:
            x_idx = int(np.random.choice(label_1_idx_list))
        flat_x = interpreter.x_dict[x_idx]
        reshape_size = (5, 72) # (10, 120)
        x_reshaped = flat_x.reshape(reshape_size) 
        label_idx = interpreter.x_label_dict[x_idx]
        flat_weights = interpreter.weights_dict[x_idx]
        weights_reshaped = flat_weights.reshape(reshape_size)
        flat_z = interpreter.z_terms[x_idx]
        z_reshaped = flat_z.reshape(reshape_size)
        annotate_matrix = np.copy(x_reshaped)
        for i in range(annotate_matrix.shape[0]):
            for j in range(annotate_matrix.shape[1]):
                if annotate_matrix[i,j] == 1.000:
                    annotate_matrix[i,j] = 1
                else:
                    annotate_matrix[i,j] = 0
        # sns.heatmap(ax=ax[ax_idx], data=weights_reshaped, annot=annotate_matrix, cmap='viridis', vmin=min_weight, vmax=max_weight, annot_kws={"fontsize":7}, cbar=False)
        sns.heatmap(ax=ax_weights[ax_idx], data=weights_reshaped, annot=annotate_matrix, cmap='viridis', annot_kws={"fontsize":7}, cbar=True)
        ax_weights[ax_idx].set_title(f'Helix instance {x_idx}. {interpreter.subset.capitalize()} Label: {label_idx}')
        sns.heatmap(ax=ax_z[ax_idx], data=z_reshaped, annot=annotate_matrix, cmap='viridis', annot_kws={"fontsize":7}, cbar=True)
        ax_z[ax_idx].set_title(f'Helix instance {x_idx}. {interpreter.subset.capitalize()} Label: {label_idx}')
    fig_weights.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.25)
    fig_z.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.25)
    fig_weights.savefig(f'{results_plots}{interpreter.name}_interpretability_weights_{len(helix_idx_list)}.pdf')
    fig_z.savefig(f'{results_plots}{interpreter.name}_interpretability_z_{len(helix_idx_list)}.pdf')

# Implement the image for plotting the weights of each timestepm

def plotTimeSerieWeights(feat,weights,ax,w_range=None,linewidth=5.,cmap='viridis'):
    """
    PARAMS
    feat (array)   -->  input time serie. Dimensions [N,1]
    weights     -->  weigths of importance encoded by color.  Dimensions [N,1]
    ax          -->  matplotlib axes where the the colored time serie will be plotted

    RETURNS
    none
    """
    N,_ = weights.shape
    if w_range == None:
        w_range = (weights.min(),weights.max())
    x = np.arange(N).reshape((-1,1))
    points = np.hstack([x,feat]).reshape((-1,1,2))
    segments = np.concatenate([points[:-1],points[1:]],axis=1)
    # lc = LineCollection(segments,cmap=plt.get_cmap(cmap),
    #     norm=plt.Normalize(w_range[0], w_range[1]))

    lc = LineCollection(segments,cmap=plt.get_cmap(cmap))

    lc.set_array((weights.squeeze()-w_range[0])/w_range[1])
    lc.set_linewidth(linewidth)
    ax.add_collection(lc)

def plotMultivariateWeihts(interpreter,n,name = "Explanations",scale=None,linewidth=9.,cmap='viridis'):
    # PARAMS
    # interpreter    --> Object interpreter
    # n              --> Number of examples to be plotted
    # RETURN
    # Plot nxn samples colored by the weights unwrapped  

    idxs = np.random.permutation(100)[:n*n]

    fig = plt.figure(name, figsize=(16,9))

    scale = interpreter.w_range
    for i,idx in enumerate(idxs):
        if interpreter.model.model_type == 'relu-mlp':
            timeserie = interpreter.x_dict[idx].reshape(5,72).T
            weights    = interpreter.weights_dict[idx].reshape(5,72).T
        elif interpreter.model.model_type == 'conv-relu-mlp':
            timeserie = interpreter.x_dict[idx].squeeze()
            weights   = interpreter.z_terms[idx].squeeze()
        label_pred     = interpreter.y_pred[idx]
        #weights    = np.power(weights,2)
        label      = interpreter.label[idx]
        ax = fig.add_subplot(n,n,i+1)            
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks([])
        ax.set_title(f"Idx: {idx} Label: {label} Prediction: {label_pred}")
        
        if scale == 'local':
            scale = (np.min(weights),np.max(weights))


        for m in range(5):
            m_timeserie = timeserie[:,m].reshape(-1,1)
            m_weights  = weights[:,m].reshape(-1,1)
            plotTimeSerieWeights( m_timeserie+ (1.4*m),m_weights,ax,scale,linewidth,cmap)
        
        plt.xlim((0,72))
        plt.ylim((-0.1,(1.4*5)))
        plt.tight_layout(pad=0,h_pad=0.1)

# for data_str in datasets:
#     if 'helix' not in data_str:
#         interpreter = load_obj(results_obj, f'{data_str}_interpreter_{len(idx_list)}.pkl')
#         scatter_2d(interpreter)

# helix_idx_list = [0, 1, 2, 3, 4, 5, 6, 7]
# for data_str in datasets:
#     if 'helix' in data_str:
#         plot_helix(helix_idx_list, data_str)
        # interpreter = load_obj(results_obj, f'{data_str}_interpreter_{len(idx_list)}.pkl')
        # plot_helix_interpretability(interpreter, 8)