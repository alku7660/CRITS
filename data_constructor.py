import numpy as np
import pandas as pd
import os
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sktime.datasets import load_from_tsfile
from address import dataset_dir

class Dataset:

    def __init__(self, dataset_name, train_fraction, seed, n=1000) -> None:
        self.name = dataset_name
        self.train_fraction = train_fraction
        self.seed = seed
        self.n = n
        self.n_timesteps         = 0
        self.n_feats             = 0
        self.train, self.train_target, self.test, self.test_target = self.load_dataset()

    def helix(self):
        dataset = pd.read_csv(f'{dataset_dir}/helix/{self.name}.csv', index_col=0)
        dataset, target = dataset.iloc[:,:-1].values, dataset['label'].values
        self.n_feats=5
        self.n_timesteps= 72
        return dataset, target
    def blink_EEG(self):
        dataset = pd.read_csv(f'{dataset_dir}/EEG-Blink-dataset/{self.name}.csv', index_col=0)
        dataset,target = dataset.iloc[:,:-1].values,dataset.iloc[:,-1].values 
        self.n_feats=3
        self.n_timesteps=510
        return dataset, target
    def ucr_datasets(self):

        def normalize_features(X_train, X_test):
            number_features = X_train.shape[1]
            for feature_idx in range(number_features):
                scaler = MinMaxScaler(clip = True)
                scaler.fit(X_train[:,feature_idx,:])
                X_train[:,feature_idx,:] = scaler.transform(X_train[:,feature_idx,:])
                X_test[:,feature_idx,:] = scaler.transform(X_test[:,feature_idx,:])
            return X_train, X_test

        def binarize_label(y_train, y_test):
            binarizer = LabelBinarizer()
            binarizer.fit(y_train)
            y_train = binarizer.transform(y_train)
            y_test = binarizer.transform(y_test)
            return y_train, y_test

        MV_PATH = f'{dataset_dir}'
        dataset = self.name        
        data = {}
        data["TRAIN"], data["TEST"] = {}, {}
        data_path = os.path.join(MV_PATH, dataset)
        data["TRAIN"]["X"], data["TRAIN"]["y"] = load_from_tsfile(
            os.path.join(data_path, dataset+"_TRAIN.ts"),  return_data_type="numpy3d"
        )
        data["TEST"]["X"], data["TEST"]["y"] = load_from_tsfile(
            os.path.join(data_path, dataset+"_TEST.ts"),  return_data_type="numpy3d"
        )
        data["TRAIN"]["X"], data["TEST"]["X"] = normalize_features(data["TRAIN"]["X"], data["TEST"]["X"])
        data["TRAIN"]["y"], data["TEST"]["y"] = binarize_label(data["TRAIN"]["y"], data["TEST"]["y"])
        if self.name == 'SelfRegulationSCP1':
            self.n_feats        = 6
            self.n_timesteps    = 896
        elif self.name == 'Heartbeat':
            self.n_feats        = 61
            self.n_timesteps    = 405
        elif self.name == 'GunPointMaleVersusFemale':
            self.n_feats        = 1
            self.n_timesteps    = 150
        elif self.name == 'SharePriceIncrease':
            self.n_feats        = 1
            self.n_timesteps    = 60
        elif self.name == 'Strawberry':
            self.n_feats        = 1
            self.n_timesteps    = 235      
        return data

    def load_dataset(self):
        np.random.seed(self.seed)
        if self.name == 'circles':
            dataset, target = make_circles(n_samples=self.n, factor=0.5, noise=0.1, random_state=self.seed)
            train, test = dataset[:int(self.train_fraction*len(dataset)),:], dataset[int(self.train_fraction*len(dataset)):,:]
            train_target, test_target = target[:int(self.train_fraction*len(target))], target[int(self.train_fraction*len(target)):]
        elif self.name == 'moons':
            dataset, target = make_moons(n_samples=self.n, noise=0.1, random_state=self.seed)
            train, test = dataset[:int(self.train_fraction*len(dataset)),:], dataset[int(self.train_fraction*len(dataset)):,:]
            train_target, test_target = target[:int(self.train_fraction*len(target))], target[int(self.train_fraction*len(target)):]
        elif 'Sine' in self.name:
            dataset, target = self.helix()
            train, test = dataset[:int(self.train_fraction*len(dataset)),:], dataset[int(self.train_fraction*len(dataset)):,:]
            train_target, test_target = target[:int(self.train_fraction*len(target))], target[int(self.train_fraction*len(target)):]
        elif 'blink_EEG' in self.name:
            dataset, target = self.blink_EEG()
            train, test = dataset[:int(self.train_fraction*len(dataset)),:], dataset[int(self.train_fraction*len(dataset)):,:]
            train_target, test_target = target[:int(self.train_fraction*len(target))], target[int(self.train_fraction*len(target)):]
        elif self.name in ['Heartbeat','SelfRegulationSCP1','GunPointMaleVersusFemale','SharePriceIncrease','Strawberry']:
            data = self.ucr_datasets()
            train, train_target, test, test_target = data["TRAIN"]["X"], data["TRAIN"]["y"], data["TEST"]["X"], data["TEST"]["y"]
            train = train.reshape((train.shape[0], int(self.n_feats*self.n_timesteps)))
            test = test.reshape((test.shape[0], int(self.n_feats*self.n_timesteps)))
        return train, train_target, test, test_target
        