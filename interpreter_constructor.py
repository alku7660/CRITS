import numpy as np
import ipdb
import tensorflow as tf
from utils import SmoothGrad
import shap

class Interpreter:
    
    def __init__(self, data, model, idx_list, subset='test',use_bias=True) -> None:
        self.name = data.name
        self.train = data.train
        self.test = data.test
        self.n_timesteps = data.n_timesteps
        self.n_feats = data.n_feats
        self.model = model
        self.idx_list = idx_list
        self.subset = subset
        self.use_bias = use_bias
        self.label = data.train_target if subset == 'train' else data.test_target

        # TODO: explainer as classes 
        self.explainers = {"CRITS":self.conv_relu_mlp_interpretation,
                           "Rectifier": self.mlp_relu_interpretation,
                           "CAM": self.CAM_conv_interpretation,
                           "SmoothGRAD":self.smothgrad_interpretation,
                           "DeepSHAP":self.deepSHAP_interpretation,
                           "GradientSHAP":self.gradientSHAP_interpretation}

    def getSparsity(self):
        # Sparsity metric: https://arxiv.org/pdf/2201.13291.pdf 
        z = [r for r in self.z_terms.values()]
        z = np.array(z)
        z = np.abs(z)
        if(z[0].shape[-1]!=self.n_feats):
            z = np.repeat(z,self.n_feats,axis=-1)

        sparsity = np.sum(z>0.1,axis=(1,2))/(self.n_feats*self.n_timesteps)
        return np.mean(sparsity)
        
    def mlp_relu_interpretation(self,x):
        x_ = np.copy(x)
        x_weights, x_intercept = self._mlp_relu_interpretation(x_)
        z = x_.reshape(x_weights.shape)*x_weights
        if self.use_bias:
            z = z + np.ones_like(z)*(x_intercept/360)
        z = z.reshape((self.n_feats,self.n_timesteps)).T
        x_weights = x_weights.reshape((self.n_feats,self.n_timesteps)).T

        return x_weights, x_intercept,z


    def _mlp_relu_interpretation(self, x):
        w_list, b_list = self.model.coefs_, self.model.intercepts_
        
        layers = range(len(w_list))
        layer_j = np.copy(x)
        activation_pattern = []
        for j in layers:
            if j == len(w_list) - 1:
                continue
            layer_j = layer_j @ w_list[j] + b_list[j]
            layer_j[layer_j <= 0] = 0
            activation_pattern.extend(np.where(layer_j > 0))
        
        for j in layers:
            if j == 0:
                feature_weights_layer = w_list[j][:,activation_pattern[j]]
                intercepts_layer = b_list[j][activation_pattern[j]]
            elif j > 0 and j < len(w_list) - 1:
                layer_j_w_active_input = w_list[j][activation_pattern[j-1],:]
                layer_j_w_active_output = layer_j_w_active_input[:,activation_pattern[j]]
                feature_weights_layer = feature_weights_layer @ layer_j_w_active_output
                intercepts_layer = intercepts_layer @ layer_j_w_active_output + b_list[j][activation_pattern[j]]
            elif j == len(w_list) - 1:
                layer_j_w_active_input = w_list[j][activation_pattern[j-1],:]
                feature_weights_layer = feature_weights_layer @ layer_j_w_active_input
                intercepts_layer = intercepts_layer @ layer_j_w_active_input + b_list[j]
        return feature_weights_layer, intercepts_layer

    def CAM_conv_interpretation(self,x):
        T = x.shape[1] 
        F,w_ = self.model.get_feature_map_weights_CAM(x)
        z = np.zeros((F.shape[0],1))
        for map in range(len(w_)):
            z = z + (F[...,map]*w_[map]).reshape((-1,1))
        
        return np.repeat(z,int(T/len(z)),axis=0)


    def conv_relu_mlp_interpretation(self,x,percent_act_info=1,onlyPosVals=False):
        
        x_ = np.copy(x)
        #x_ = x_.reshape(1,self.n_timesteps,1,self.n_feats)
        maxPos,maxVal,featMap, weights,deConv = self.model.getConvInfo(x_)
        # Max positions
        maxPos = np.argmax(featMap,axis=0) 
        # Extracting the dense weights 
        denseWeights,bias = self._mlp_relu_interpretation(maxVal)
        denseWeights =denseWeights.squeeze()
        #denseWeights = self.model.getDenseInfo(x) #TODO: change this for the unwrap algorithm
        # Filtering zero-activated neurons
        denseWeights = denseWeights*(maxVal>0)
        if onlyPosVals:
            denseWeights[denseWeights<0] = 0
        # Computing amount of filters needed to achive percent_act_info
        weights_actMaxVal = np.abs(maxVal*denseWeights)
        total_val_weights = np.sum(weights_actMaxVal)
        total_val_weights_percentage = total_val_weights*percent_act_info
        sorted_val_weights = np.sort(weights_actMaxVal)
        sorted_val_weights = sorted_val_weights[::-1]
        sorted_val_weights_cum_sum = np.cumsum(sorted_val_weights)
        # TODO: replace the loop for a vectorized expression
        for n in range(len(sorted_val_weights_cum_sum)):
            if sorted_val_weights_cum_sum[n] >= total_val_weights_percentage:
                break
        idx_top = np.argsort(weights_actMaxVal)[-n:]
        topPos = maxPos[idx_top]
        topVals = maxVal[idx_top]

        expl = np.zeros((self.n_timesteps,self.n_feats))
        for i in range(n):
            mask = np.zeros((1,deConv.input.shape[1],deConv.input.shape[2],deConv.input.shape[3]))
            mask[0,topPos[i],0,idx_top[i]] = 1
            deConv_values = deConv.predict(mask).reshape(self.n_timesteps, self.n_feats)
            expl = expl + denseWeights[idx_top[i]]*deConv_values
        
        x_weights = expl
        x_intercept = bias
        
        x_ = x_.reshape(self.n_timesteps, self.n_feats)
        z = x_weights*x_
        if self.use_bias:
            z = z + np.ones_like(z)*(x_intercept/(self.n_feats*self.n_timesteps)) #TODO: set 360 to a variable 


        return x_weights,x_intercept,z

    def smothgrad_interpretation(self,x):
        # SmoothGRAD explainer using tf-explain library: https://github.com/sicara/tf-explain
        x_ = np.copy(x)
        #ipdb.set_trace()
        smthGRAD_model = SmoothGrad()
        x_weights,x_intercept,z = np.zeros_like(x_),np.zeros_like(x_),np.zeros_like(x_)
        y_pred = self.model.classifier.predict(x)[0,0]
        z = smthGRAD_model.explain(validation_data=(x_,None),
                                   model = self.model.classifier,
                                   class_index = y_pred,
                                   num_samples = 20,
                                   noise = 0.2)
        z = z.numpy().reshape(self.n_timesteps, self.n_feats)
        z = (z-np.nanmin(z))/(np.nanmax(z)-np.nanmin(z))
        return x_weights,x_intercept,z


    def gradientSHAP_interpretation(self,x):
        # DeepSHAP explainer using SHAP library: https://shap-lrjball.readthedocs.io/en/latest/generated/shap.DeepExplainer.html
        x_ = np.copy(x)
        
        
        x_weights,x_intercept,z = np.zeros_like(x_),np.zeros_like(x_),np.zeros_like(x_)

        background = np.copy(self.train)
        background = background.reshape((-1,self.n_feats,self.n_timesteps))
        background = np.transpose(background,(0,2,1)).reshape(-1,self.n_timesteps,1,self.n_feats)
                        
        gradSHAP_model = shap.GradientExplainer(self.model.classifier_SHAP, background)
        
        y_pred = self.model.classifier.predict(x)[0,0]
        z = gradSHAP_model.shap_values(x_)
        z = np.array(z).reshape(self.n_timesteps, self.n_feats)
        #z = (z-np.nanmin(z))/(np.nanmax(z)-np.nanmin(z))
        return x_weights,x_intercept,z


    def deepSHAP_interpretation(self,x,background_samples=100):
        # DeepSHAP explainer using SHAP library: https://shap-lrjball.readthedocs.io/en/latest/generated/shap.DeepExplainer.html
        x_ = np.copy(x)
        
        
        x_weights,x_intercept,z = np.zeros_like(x_),np.zeros_like(x_),np.zeros_like(x_)

        background = np.copy(self.train[np.random.choice(self.train.shape[0], background_samples, replace=False)])
        background = background.reshape((-1,self.n_feats,self.n_timesteps))
        background = np.transpose(background,(0,2,1)).reshape(-1,self.n_timesteps,1,self.n_feats)
        
        deepSHAP_model = shap.DeepExplainer(self.model.classifier_SHAP, background)
        
        y_pred = self.model.classifier.predict(x)[0,0]
        z = deepSHAP_model.shap_values(x_)
        z = np.array(z).reshape(self.n_timesteps, self.n_feats)
        #z = (z-np.nanmin(z))/(np.nanmax(z)-np.nanmin(z))
        return x_weights,x_intercept,z



    def interpret_instances(self, explainer, X=None):
        """
        It extracts all the saliency/relevance maps.

        PARAMS
        ----------------
        explainer [String]  --> Saliency/relevance method to be applied
        X [numpy array]     --> array of instances to get the explanations for (None if the method is to explain train/test subsets)

        RETURN
        ----------------
        x_dict          [ndarray]      -->  Time series evaluated 
        x_label_dict    [ndarray]      -->  Labels for time series under study 
        y_pred          [ndarray]      -->  Predicted labels for time series under study 
        weights_dict    [ndarray]      -->  Weights (w_t) of each time step
        intercept_dict  [ndarray]      -->  Bias array (b_t). Only for CRITS.
        z_terms         [ndarray]      -->  Relevance terms (r_t). Only for CRITS.

        """
        if  not (explainer in self.explainers):
            raise Exception(f"{explainer} explainer is not available")
        if self.explainers[explainer] == None:
            raise Exception(f"{explainer} explainer is not implemented yet")
        if not( explainer in self.model.explainers):
            raise Exception(f"{explainer} is not compatible for model {self.model.model_type}")
        
        explainer_method = self.explainers[explainer]
        if X == None:
            x_dict, x_label_dict, y_pred, weights_dict, intercept_dict, z_terms = self._interpret_instances(explainer_method)
        else:
            x_dict, x_label_dict, y_pred, weights_dict, intercept_dict, z_terms = self._interpret_instances_specific(explainer_method, X)

        w_range = (np.min([w for _,w in weights_dict.items()]),np.max([w for _,w in weights_dict.items()]))
        z_range = (np.nanmin([z for _,z in z_terms.items()]),np.nanmax([w for _,w in z_terms.items()]))
        
        return x_dict, x_label_dict, y_pred, weights_dict, intercept_dict, z_terms, w_range, z_range


    def _interpret_instances(self, explainer_method):
        all_x, all_labels,all_pred, all_weights, all_intercepts, z_terms = {}, {}, {}, {}, {},{}
        for idx in self.idx_list:
            label = self.label[idx]
            if self.subset == 'train':
                x = self.train[idx]
            else:
                x = self.test[idx]

            if self.model.model_type == 'relu-mlp':
                x_weights, x_intercept,z = explainer_method(x)                
                y_pred = self.model.classifier.predict(x.reshape(1,-1))[0]
                x = x.reshape((self.n_feats,self.n_timesteps)).T
            
            elif self.model.model_type == 'conv-relu-mlp':
                x = x.reshape((-1,self.n_feats,self.n_timesteps))
                x = np.transpose(x,(0,2,1)).reshape(-1,self.n_timesteps,1,self.n_feats)
                x_weights, x_intercept, z = explainer_method(x)
                y_pred = self.model.classifier.predict(x)[0]
                x = x.reshape((self.n_timesteps,self.n_feats))

            elif self.model.model_type == 'CAM-conv':
                x = x.reshape((-1,self.n_feats,self.n_timesteps))
                x = np.transpose(x,(0,2,1)).reshape(-1,self.n_timesteps,1,self.n_feats)
                z = explainer_method(x)
                y_pred = self.model.classifier.predict(x)[0]
                x = x.squeeze()
                x_weights = 0
                x_intercept =0

            all_x[idx], all_labels[idx],all_pred[idx], all_weights[idx], all_intercepts[idx], z_terms[idx] = x, label,y_pred, x_weights, x_intercept, z      
        
        return all_x, all_labels, all_pred, all_weights, all_intercepts, z_terms
    
    def _interpret_instances_specific(self, explainer, X):
        explainer_method = self.explainers[explainer]
        all_x, all_pred, all_weights, all_intercepts, z_terms = {}, {}, {}, {}, {}
        idx_list = list(range(X.shape[0]))
        for idx in self.idx_list:
            x = X[idx]

            if self.model.model_type == 'relu-mlp':
                x_weights, x_intercept,z = explainer_method(x)                
                y_pred = self.model.classifier.predict(x.reshape(1,-1))[0]
                x = x.reshape((self.n_feats,self.n_timesteps)).T
            
            elif self.model.model_type == 'conv-relu-mlp':
                x = x.reshape((-1,self.n_feats,self.n_timesteps))
                x = np.transpose(x,(0,2,1)).reshape(-1,self.n_timesteps,1,self.n_feats)
                x_weights, x_intercept, z = explainer_method(x)
                y_pred = self.model.classifier.predict(x)[0]
                x = x.squeeze()

            elif self.model.model_type == 'CAM-conv':
                x = x.reshape((-1,self.n_feats,self.n_timesteps))
                x = np.transpose(x,(0,2,1)).reshape(-1,self.n_timesteps,1,self.n_feats)
                z = explainer_method(x)
                y_pred = self.model.classifier.predict(x)[0]
                x = x.squeeze()
                x_weights = 0
                x_intercept =0

            all_x[idx], all_pred[idx], all_weights[idx], all_intercepts[idx], z_terms[idx] = x, y_pred, x_weights, x_intercept, z      
        
        return all_x, all_pred, all_weights, all_intercepts, z_terms