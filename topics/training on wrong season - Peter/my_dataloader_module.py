#python file to collect all classes and functions related to dataloader generation
import pandas as pd 
from pandas import Series, DataFrame 
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn.model_selection import train_test_split
import xarray as xr


class TimeseriesDatasetPipeline:
    """Pipeline to convert xarray Dataset to PyTorch Dataset with optional scaling."""
    
    def __init__(self, ds,lag_selection=None, forecast_horizon=24,forecast_hours=1,forecast_var=0):
        """
        Initialize the pipeline.
        
        Args:
            ds: xarray Dataset containing the time series data.
            lag_selection: List of time lags to include in the input (e.g., [0, 24, 48] for current, 24h ago, and 48h ago).
            forecast_horizon: Number of time steps ahead to forecast (e.g., 24 for 24 hours ahead).
            forecast_hours: Number of hours to forecast in the end of the horizon(e.g., 1 for predicting the last entry of the horizon).
            forecast_var: Index of the variable to forecast (e.g., 0 for the first variable in the dataset).
        """
        self.ds = ds
        self.lag_selection = lag_selection
        self.forecast_horizon = forecast_horizon
        self.forecast_var = forecast_var
        self.forecast_hours = forecast_hours
        self.data = None
        self._prepare_data()
        if self.forecast_var >= self.data.shape[1]:
            raise ValueError(f"forecast_var index {self.forecast_var} is out of bounds for data with {self.data.shape[1]} variables.")
    

    def get_input_shape(self):
        """Return the shape of the input data (time, variables, space)."""
        return self.ds.to_array().values.shape if self.ds is not None else None

    def _prepare_data(self):
        """Convert xarray to numpy, flatten spatial dimensions and reshape."""
        stacked = self.ds.to_array()
        self.data = stacked.values.transpose(1, 0, 2,3)#(time, variables, lat,lon)
        print("Data prepared with shape:", self.data.shape)


    def scale_X(self,shaped_data,scaler_set=None):
        """Apply each scaler to the corresponding variable."""
        scaled=shaped_data
        if scaler_set is None:  sc_set=np.array([])
        else: sc_set=scaler_set
        for i in range(scaled.shape[-3]):#in vars
            reshaped_var = scaled[:,:, i, :].reshape(-1, 1)
            if scaler_set is None:#scaler set not functional
                sc_set = np.append(sc_set, sklearn.preprocessing.StandardScaler().fit(reshaped_var))
            sh=scaled.shape[0], scaled.shape[1],scaled.shape[3], scaled.shape[4]
            scaled[:,:, i, :] = sc_set[i].transform(reshaped_var).reshape(sh)
        return scaled, sc_set
   
    def scale_y(self,shaped_data,scaler=None):
        """Apply scaler to the corresponding variable."""
        scaled=shaped_data
        reshaped_var = scaled.reshape(-1, 1)
        if scaler is None:
            scaler=sklearn.preprocessing.StandardScaler().fit(reshaped_var)
        sh=scaled.shape[0],1,scaled.shape[2], scaled.shape[3]
        scaled = scaler.transform(reshaped_var).reshape(sh)
        return scaled,scaler
    



    def create_windowed_dataset_from_idxlist(self, idx_list):
        """Create X and y using moving window method."""
        data = self.data
        if self.lag_selection:
            max_lag = max(self.lag_selection)+1
        else:
            raise ValueError("forecast_lag_selection must be provided.")
        
        X, y = [], []
        idx_skipped=[]
        for i in idx_list:
            #print(f"Processing index {i} with lag selection {self.lag_selection} and forecast horizon {self.forecast_horizon}.")
            #print(f"\nforecast horizon: {self.forecast_horizon}, forecast var: {self.forecast_var}, lag selection: {self.lag_selection}")
            if i+max_lag+self.forecast_horizon >= len(data):
                #print(f"Skipping index {i} due to insufficient data for lag selection and forecast horizon.")
                idx_skipped.append(i)
                continue
            #testing, if there is enough consecutive data for the selected lags and forecast horizon in the time dimension, if not, skip this index
            last_timespan_idx=i+max_lag+self.forecast_horizon
            if self.ds.time[last_timespan_idx].values - self.ds.time[i].values > np.timedelta64(max_lag+self.forecast_horizon, 'h'):
                #print(f"Skipping index {i} due to insufficient consecutive data for forecast horizon.")
                idx_skipped.append(i)
                continue
            
            #print("Original data shape:", data.shape)
            data_filtered = []#filtering data based on lag selection
            for t in self.lag_selection:
                data_filtered.append(data[i+t,:,:])
            xd = np.array(data_filtered)  # (lag_selection, variables, space)
            #print("data_filtered:", getattr(xd, "shape", None))
            
            #xd=np.delete(data_filtered, self.forecast_var, axis=1)# Remove target variable from input
            #print("xd:", getattr(xd, "shape", None), type(xd),"\n")
            X.append(xd)
            yd=data[i+self.forecast_horizon,self.forecast_var,:]  # Only target variable
            yd=yd.reshape(1, yd.shape[0], yd.shape[1])
            y.append(yd)
            #print("yd:", getattr(yd, "shape", None), type(yd))

        X = np.array(X) 
        y = np.array(y)
        print(f"skipped number of indices: {len(idx_skipped)} out of {len(idx_list)} total indices.") 
        print("X:", X.shape, type(X))   # (samples, lag, variables, space)
        print("y:",y.shape, type(y))    # (samples, horizon, pred variable, space)
        return X,y,idx_skipped


    def create_random_train_test_split_idx(self, test_size=0.2, random_state=None):
        """Create random train/test split indices."""
        total_samples = self.data.shape[0] - max(self.lag_selection) - self.forecast_horizon
        indices = np.arange(total_samples)
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
        return train_idx, test_idx
    

    def to_scaled_pytorch_train_test_dataloaders(self, batch_size=32, test_size=0.2, random_state=None):
        """Create PyTorch DataLoader with random train/test split."""
        train_idx, test_idx = self.create_random_train_test_split_idx(test_size=test_size, random_state=random_state)
        X_train, y_train, skipped_idx_list_train= self.create_windowed_dataset_from_idxlist(train_idx)
        X_test, y_test, skipped_idx_list_test = self.create_windowed_dataset_from_idxlist(test_idx)
        
        #scale
        X_train_sc, scaler_set = self.scale_X(X_train)
        X_test_sc, _ = self.scale_X(X_test, scaler_set=scaler_set)
        y_train_sc, scaler_y = self.scale_y(y_train)
        y_test_sc, _ = self.scale_y(y_test, scaler=scaler_y)

        train_dataset = TimeseriesPyTorchDataset(X_train_sc, y_train_sc)
        test_dataset = TimeseriesPyTorchDataset(X_test_sc, y_test_sc)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, test_loader, skipped_idx_list_train, skipped_idx_list_test, scaler_set, scaler_y
    



    def to_pytorch_dataloader(self, batch_size=32, shuffle=True):
        """Convert to PyTorch DataLoader."""
        X, y, skipped_idx_list = self.create_windowed_dataset_from_idxlist(list(range(self.data.shape[0] - max(self.lag_selection) - self.forecast_horizon)))
        print("dataloader not scaled!")
        dataset = TimeseriesPyTorchDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), skipped_idx_list


    def to_scaled_pytorch_dataloader(self, batch_size=32, shuffle=True,scaler_set=None, scaler_y=None):
        """Convert to scaled PyTorch DataLoader."""
        X, y, skipped_idx_list = self.create_windowed_dataset_from_idxlist(list(range(self.data.shape[0] - max(self.lag_selection) - self.forecast_horizon)))
        #print("creating scaled dataloader...")
        X_scaled, scaler_set = self.scale_X(X, scaler_set=scaler_set)
        #print("X scaled with scaler set:", scaler_set)
        y_scaled, scaler_y = self.scale_y(y, scaler=scaler_y)
        #print("y scaled with scaler y:", scaler_y)
        dataset = TimeseriesPyTorchDataset(X_scaled, y_scaled)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), skipped_idx_list, scaler_set, scaler_y



class TimeseriesPyTorchDataset(Dataset):
    """PyTorch Dataset for timeseries data."""
    
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
