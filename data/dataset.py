import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os


class StreamerDischargeDataset(Dataset):
    """Dataset for streamer discharge simulation data"""
    
    def __init__(self, h5_file_path, input_key='N2_C3', output_key='electric_fld', 
                 transform=None, normalize=True, normalization_type='standard', train=True, train_ratio=0.8):
        """
        Args:
            h5_file_path (str): Path to the HDF5 file
            input_key (str): Key for input data (default: 'N2_C3')
            output_key (str): Key for output data (default: 'electric_fld')
            transform (callable, optional): Optional transform to be applied on a sample
            normalize (bool): Whether to normalize the data
            normalization_type (str): Type of normalization ('standard' or 'minmax')
            train (bool): Whether this is training data (True) or validation data (False)
            train_ratio (float): Ratio of data to use for training
        """
        self.h5_file_path = h5_file_path
        self.input_key = input_key
        self.output_key = output_key
        self.transform = transform
        self.normalize = normalize
        self.normalization_type = normalization_type
        self.train = train
        self.train_ratio = train_ratio
        
        # Load data
        self._load_data()
        
        # Split data into train/validation
        self._split_data()
        
        # Normalize data if requested
        if self.normalize:
            self._normalize_data()
    
    def _load_data(self):
        """Load data from HDF5 file"""
        with h5py.File(self.h5_file_path, 'r') as f:
            self.input_data = f[self.input_key][:]  # Shape: (n_samples, nx, ny)
            self.output_data = f[self.output_key][:]  # Shape: (n_samples, nx, ny)
        
        print(f"Loaded data shapes - Input: {self.input_data.shape}, Output: {self.output_data.shape}")
    
    def _split_data(self):
        """Split data into training and validation sets"""
        total_samples = len(self.input_data)
        train_size = int(total_samples * self.train_ratio)
        
        if self.train:
            self.input_data = self.input_data[:train_size]
            self.output_data = self.output_data[:train_size]
        else:
            self.input_data = self.input_data[train_size:]
            self.output_data = self.output_data[train_size:]
        
        print(f"{'Training' if self.train else 'Validation'} data size: {len(self.input_data)}")
    
    def _normalize_data(self):
        """Normalize input and output data"""
        # Choose scaler based on normalization type
        if self.normalization_type == 'minmax':
            input_scaler = MinMaxScaler()
            output_scaler = MinMaxScaler()
            scaler_name = "MinMaxScaler"
        else:  # default to standard
            input_scaler = StandardScaler()
            output_scaler = StandardScaler()
            scaler_name = "StandardScaler"
        
        # Normalize input data
        self.input_scaler = input_scaler
        input_reshaped = self.input_data.reshape(-1, self.input_data.shape[-1] * self.input_data.shape[-2])
        input_normalized = self.input_scaler.fit_transform(input_reshaped)
        self.input_data = input_normalized.reshape(self.input_data.shape)
        
        # Normalize output data
        self.output_scaler = output_scaler
        output_reshaped = self.output_data.reshape(-1, self.output_data.shape[-1] * self.output_data.shape[-2])
        output_normalized = self.output_scaler.fit_transform(output_reshaped)
        self.output_data = output_normalized.reshape(self.output_data.shape)
        
        print(f"Data normalized using {scaler_name}")
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_sample = self.input_data[idx]
        output_sample = self.output_data[idx]
        
        # Convert to torch tensors and add channel dimension
        input_sample = torch.FloatTensor(input_sample).unsqueeze(0)  # Shape: (1, nx, ny)
        output_sample = torch.FloatTensor(output_sample).unsqueeze(0)  # Shape: (1, nx, ny)
        
        if self.transform:
            input_sample = self.transform(input_sample)
            output_sample = self.transform(output_sample)
        
        return input_sample, output_sample


def get_data_loaders(h5_file_path, batch_size=32, num_workers=4, train_ratio=0.8, 
                    input_key='N2_C3', output_key='electric_fld', normalize=True, normalization_type='standard'):
    """
    Create training and validation data loaders
    
    Args:
        h5_file_path (str): Path to the HDF5 file
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        train_ratio (float): Ratio of data to use for training
        input_key (str): Key for input data
        output_key (str): Key for output data
        normalize (bool): Whether to normalize the data
        normalization_type (str): Type of normalization ('standard' or 'minmax')
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = StreamerDischargeDataset(
        h5_file_path=h5_file_path,
        input_key=input_key,
        output_key=output_key,
        train=True,
        train_ratio=train_ratio,
        normalize=normalize,
        normalization_type=normalization_type
    )
    
    val_dataset = StreamerDischargeDataset(
        h5_file_path=h5_file_path,
        input_key=input_key,
        output_key=output_key,
        train=False,
        train_ratio=train_ratio,
        normalize=normalize,
        normalization_type=normalization_type
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    h5_path = os.path.join(os.path.expanduser("~"), "projects/data/dataset_128.h5")
    if os.path.exists(h5_path):
        train_loader, val_loader = get_data_loaders(h5_path, batch_size=4)
        
        # Test a batch
        for batch_idx, (inputs, outputs) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Input shape: {inputs.shape}, Output shape: {outputs.shape}")
            if batch_idx >= 2:  # Test only first 3 batches
                break
    else:
        print(f"Dataset file not found at {h5_path}")
