import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from models.unet import UNet
from data.dataset import get_data_loaders


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(f"runs/{config['experiment_name']}")
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Train/Loss_Batch', loss.item(), 
                                     len(self.train_losses) * len(self.train_loader) + batch_idx)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.config['output_dir'], 'latest_checkpoint.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.config['output_dir'], 'best_checkpoint.pth'))
            print(f"New best model saved at epoch {epoch}")
    
    def plot_sample_predictions(self, epoch, num_samples=4):
        """Plot sample predictions"""
        self.model.eval()
        with torch.no_grad():
            # Get a batch from validation set
            inputs, targets = next(iter(self.val_loader))
            inputs = inputs.to(self.device)
                
            # Get predictions
            outputs = self.model(inputs)
            
            # Move to CPU for plotting
            inputs = inputs.cpu().numpy()
            outputs = outputs.cpu().numpy()
            
            # Get target array
            targets = targets.numpy()
            
            # Create subplots
            _, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
            
            for i in range(num_samples):
                # Input (N2_C3)
                axes[0, i].imshow(inputs[i, 0], cmap='plasma')
                axes[0, i].set_title(f'Input (N2_C3) - Sample {i+1}')
                axes[0, i].axis('off')
                
                # Target (electric field)
                axes[1, i].imshow(targets[i, 0], cmap='viridis')
                axes[1, i].set_title(f'Target (Electric Field) - Sample {i+1}')
                axes[1, i].axis('off')
                
                # Prediction
                axes[2, i].imshow(outputs[i, 0], cmap='viridis')
                axes[2, i].set_title(f'Prediction (Electric Field) - Sample {i+1}')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['output_dir'], f'predictions_epoch_{epoch}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log to tensorboard
            self.writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
            self.writer.add_scalar('Val/Loss_Epoch', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % self.config['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Plot sample predictions
            if epoch % self.config['plot_every'] == 0:
                self.plot_sample_predictions(epoch)
        
        # Save final model
        self.save_checkpoint(self.config['epochs']-1)
        
        # Plot training curves
        self.plot_training_curves()
        
        print("Training completed!")
        self.writer.close()
    
    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config['output_dir'], 'training_curves.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for streamer discharge mapping')
    parser.add_argument('--data_path', type=str, default=os.path.join(os.path.expanduser("~"), "projects/data/dataset_128.h5"),
                       help='Path to the HDF5 dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--input_key', type=str, default='N2_C3',
                       help='Input data key in HDF5 file')
    parser.add_argument('--output_key', type=str, default='electric_fld',
                       help='Output data key in HDF5 file')
    parser.add_argument('--normalization_type', type=str, default='standard',
                       choices=['standard', 'minmax'],
                       help='Type of normalization to use')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and plots')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--plot_every', type=int, default=20,
                       help='Plot predictions every N epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'experiment_name': f"unet_{args.input_key}_to_{args.output_key}_{args.normalization_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'input_key': args.input_key,
        'output_key': args.output_key,
        'normalization_type': args.normalization_type,
        'output_dir': args.output_dir,
        'save_every': args.save_every,
        'plot_every': args.plot_every,
        'num_workers': args.num_workers
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(
        h5_file_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_key=args.input_key,
        output_key=args.output_key,
        normalization_type=args.normalization_type
    )
    
    # Create model
    model = UNet(in_channels=1, out_channels=1, bilinear=True)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device, config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
