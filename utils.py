"""
Utility functions for HPC AI Project
Common functions used across all implementations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time

class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST classification
    Architecture: Conv -> Conv -> FC -> FC
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def get_data_loaders(batch_size=64, is_distributed=False, rank=0, world_size=1):
    """
    Create data loaders for MNIST dataset
    
    Args:
        batch_size: Batch size for training
        is_distributed: Whether using distributed training
        rank: Process rank (for distributed)
        world_size: Total number of processes
    
    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    if is_distributed:
        # For distributed training, split dataset across processes
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader

def train_epoch(model, device, train_loader, optimizer, epoch, verbose=True):
    """
    Train model for one epoch
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if verbose and batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches

def test_model(model, device, test_loader, verbose=True):
    """
    Test model accuracy
    
    Returns:
        test_loss, accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    if verbose:
        print(f'\nTest set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

def get_device_info():
    """
    Get information about available compute devices
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cpu_available': True,
        'mps_available': False,
        'cuda_available': False,
        'device_name': 'cpu'
    }
    
    if torch.backends.mps.is_available():
        info['mps_available'] = True
        info['device_name'] = 'mps'
        info['device_type'] = 'Apple Metal Performance Shaders (GPU)'
    
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['device_name'] = 'cuda'
        info['device_type'] = 'NVIDIA CUDA GPU'
    
    return info

def print_system_info():
    """Print system and PyTorch configuration"""
    print("="*70)
    print(" SYSTEM INFORMATION")
    print("="*70)
    print(f"PyTorch Version: {torch.__version__}")
    
    device_info = get_device_info()
    print(f"CPU Available: {device_info['cpu_available']}")
    print(f"MPS (Apple GPU) Available: {device_info['mps_available']}")
    print(f"CUDA Available: {device_info['cuda_available']}")
    print(f"Default Device: {device_info['device_name']}")
    
    if device_info['mps_available']:
        print(f"Using: {device_info['device_type']}")
    
    print("="*70 + "\n")

class Timer:
    """Simple timer utility"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def start(self):
        self.start_time = time.time()
        
    def stop(self):
        self.end_time = time.time()
        
    def elapsed(self):
        if self.start_time is None:
            return 0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, *args):
        self.stop()

def format_time(seconds):
    """Format seconds into human-readable string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def calculate_speedup(baseline_time, optimized_time):
    """Calculate speedup factor"""
    return baseline_time / optimized_time

def calculate_efficiency(speedup, num_processes):
    """Calculate parallel efficiency percentage"""
    return (speedup / num_processes) * 100

def print_performance_summary(name, time_taken, baseline_time=None):
    """Print formatted performance summary"""
    print(f"\n{'='*70}")
    print(f" {name}")
    print(f"{'='*70}")
    print(f"Execution Time: {format_time(time_taken)}")
    
    if baseline_time:
        speedup = calculate_speedup(baseline_time, time_taken)
        print(f"Speedup: {speedup:.2f}x")
        print(f"Time Reduction: {((baseline_time - time_taken) / baseline_time * 100):.1f}%")
    
    print(f"{'='*70}\n")

def print_banner(msg):
    print("=" * len(msg))
    print(msg)
    print("=" * len(msg))
