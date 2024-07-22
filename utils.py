import numpy as np

import tensorflow as tf

#from tf_explain.utils.display import grid_display
#from tf_explain.utils.image import transform_to_normalized_grayscale
#from tf_explain.utils.saver import save_grayscale

from sklearn.metrics import f1_score,accuracy_score,mean_squared_error


import ipdb


def sparsity_score(z):
    """
    Sparsity score

    PARAMS
    ------------------
    z           --> batch of z terms (relevance)

    RETURN
    ------------------

    sparsity    --> sparsity score 
    """
    
    z = [r for r in z.values()]
    z = np.array(z)
    z = np.abs(z)
    n_timesteps,n_feats = z.shape[1],z.shape[-1]
    if(z[0].shape[-1]!=n_feats):
        z = np.repeat(z,n_feats,axis=-1)

    sparsity = np.sum(np.abs(z)>0.01,axis=(1,2))/(n_feats*n_timesteps)
    return np.mean(sparsity)


"""
SmoothGrad  from: https://github.com/sicara/tf-explain/blob/master/tf_explain/core/smoothgrad.py
"""


class SmoothGrad:

    """
    Perform SmoothGrad algorithm for a given input

    Paper: [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)
    """

    def explain(self, validation_data, model, class_index, num_samples=5, noise=1.0):
        """
        Compute SmoothGrad for a specific class index

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            num_samples (int): Number of noisy samples to generate for each input image
            noise (float): Standard deviation for noise normal distribution

        Returns:
            np.ndarray: Grid of all the smoothed gradients
        """
        images, _ = validation_data

        noisy_images = SmoothGrad.generate_noisy_images(images, num_samples, noise)

        smoothed_gradients = SmoothGrad.get_averaged_gradients(
            noisy_images, model, class_index, num_samples
        )

        # grayscale_gradients = transform_to_normalized_grayscale(
        #     tf.abs(smoothed_gradients)
        # ).numpy()

        #grid = grid_display(grayscale_gradients)

        return smoothed_gradients

    @staticmethod
    def generate_noisy_images(images, num_samples, noise):
        """
        Generate num_samples noisy images with std noise for each image.

        Args:
            images (numpy.ndarray): 4D-Tensor with shape (batch_size, H, W, 3)
            num_samples (int): Number of noisy samples to generate for each input image
            noise (float): Standard deviation for noise normal distribution

        Returns:
            np.ndarray: 4D-Tensor of noisy images with shape (batch_size*num_samples, H, W, 3)
        """
        repeated_images = np.repeat(images, num_samples, axis=0)
        noise = np.random.normal(0, noise, repeated_images.shape).astype(np.float32)

        return repeated_images + noise

    @staticmethod
    def get_averaged_gradients(noisy_images, model, class_index, num_samples):
        """
        Compute average of gradients for target class.

        Args:
            noisy_images (tf.Tensor): 4D-Tensor of noisy images with shape
                (batch_size*num_samples, H, W, 3)
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            num_samples (int): Number of noisy samples to generate for each input image

        Returns:
            tf.Tensor: 4D-Tensor with smoothed gradients, with shape (batch_size, H, W, 1)
        """
        #num_classes = model.output.shape[1]
        
        expected_output = tf.Variable([class_index] * noisy_images.shape[0])
        expected_output = tf.reshape(expected_output,[-1,1])
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(noisy_images, tf.float32)
            tape.watch(inputs)
            predictions = model(inputs)
            loss = tf.keras.losses.binary_crossentropy(
                expected_output, predictions
            )

        grads = tape.gradient(loss, inputs)
        
        grads_per_image = tf.reshape(grads, (-1, num_samples, *grads.shape[1:]))
        averaged_grads = tf.reduce_mean(grads_per_image, axis=1)

        # Normalization
        

        return averaged_grads




def zero_perturbation(x,idx):
    x_ = np.copy(x)
    x_[idx] =0
    return x_ 


def inverse_perturbation(x,idx):
    x_ = np.copy(x)
    max_ = np.max(x_,axis=1)
    x_[idx] = max_[idx[1]] - x_[idx]
    return x_ 


def mean_perturbation(x,idxs,ns_ratio=0.1):
    x_ = np.copy(x)
    n_timesteps = x_.shape[0]
    ns = int(n_timesteps*ns_ratio)
    for i in range(len(idxs[0])):
        t = idxs[0][i]
        f = idxs[1][i]
        x_[t:t+ns,f] = np.mean(x_[t:t+ns,f]) 

    return x_ 

def swap_perturbation(x,idxs,ns_ratio=0.1):
    x_ = np.copy(x)
    n_timesteps = x_.shape[0]
    ns = int(n_timesteps*ns_ratio)
    for i in range(len(idxs[0])):
        t = idxs[0][i]
        f = idxs[1][i]
        x_[t:t+ns,f] = x_[t:t+ns,f][::-1]

    return x_ 

def random_perturbation(x, std_dev=0.01):
    """
    x = original sample(s)
    std_dev = standard deviation of the perturbation
    """
    x_ = np.copy(x)
    x_ = np.random.normal(loc=x_, scale=std_dev, size=x.shape)
    
    return x_

def pert_eval(X, y,  r, model,perc=90):
    """
    Evaluate explainers using pertubation-based metrics defined in:
        Schlegel, U., Arnout, H., El-Assady, M., Oelke, D., & Keim, D. A. (2019, October). Towards a rigorous evaluation of XAI methods on time series.
        In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW) (pp. 4197-4201). IEEE.

    
    PARAMETERS:

        x                     --> original samples
        r [numpy array]       --> relevance of input elements
        model                 --> classifier model
        perc                  --> Pertecnile used as threshold

    RETURN:
        change in perturbance for zero [point-based], inverse [point-based], swap [seq-based], mean[seq-based] perturbacions
    """

    perturbated_inputs={"zero":[],
                        "inverse":[],
                        "mean":[],
                        "swap":[]}
    perturbation_methods={"zero":zero_perturbation,
                          "inverse": inverse_perturbation,
                          "mean":mean_perturbation,
                          "swap":swap_perturbation}
    perturbation_scores = dict().fromkeys(perturbation_methods)
    performance_metric = accuracy_score

    print("Running perturbation metrics")
    thr = np.percentile(np.abs(r),perc,axis=(0,1))
    print(f"Percentiles per feature: {thr}")
    x_pert = None
    for i in range(X.shape[0]):
        print(f"Changing sample {i}/{len(X)}]")
        idx_rel = np.nonzero(np.greater(np.abs(r)[i],thr))

        for perturbation in perturbated_inputs.keys():
            method = perturbation_methods[perturbation]
            perturbated_inputs[perturbation].append(method(X[i],idx_rel))
    
    # y_ = model.predict(X)
    y_ = model.predict_proba(X)
    # performance_score = performance_metric(y,y_)
    for perturbation,data in perturbated_inputs.items():
        perturbated_inputs[perturbation] = np.array(data)
        y_pertubated_ = model.predict_proba(perturbated_inputs[perturbation])
        # perturbed_performance_score = performance_metric(y,y_pertubated_)
        #perturbation_scores[perturbation] = np.abs(perturbed_performance_score-performance_score)
        # perturbation_scores[perturbation] = performance_metric(y_,y_pertubated_)
        perturbation_scores[perturbation] = mean_squared_error(y_,y_pertubated_,squared=True)


    return  perturbated_inputs,perturbation_scores 

def explanation_sensitivity_score(interpreter, explainer, std_dev, X=None):
    """   
    PARAMETERS:

        X                     --> original samples
        interpreter           --> interpreter object that includes the trained classifier
        explainer             --> explainer string referring to the name of the explainer to use
        std_dev               --> standard deviation for the normal perturbation

    RETURN:
        Explanation-sensitivity score: Change in the explanations weights over total change in the instance
    """
    if X == None:
        X = interpreter.test
    rmse_weights = {}
    X_perturbed = random_perturbation(X, std_dev=std_dev)
    all_x, all_pred, all_weights, all_intercepts, all_z = interpreter._interpret_instances_specific(explainer, X=X)
    pert_x, pert_pred, pert_weights, pert_intercepts, pert_z = interpreter._interpret_instances_specific(explainer, X=X_perturbed)
    for pert_z_idx, pert_z_x in pert_z.items():
        all_z_x = all_z[pert_z_idx]
        rmse_weights[pert_z_idx] = np.linalg.norm(pert_z_x - all_z_x)/np.sqrt(all_z_x.size)
    return rmse_weights
    




