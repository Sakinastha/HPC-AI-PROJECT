import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import print_banner  # or other functions you need

"""
MPI Distributed Training - Data Parallelism
HPC AI Project: Multi-process distributed training using MPI

This demonstrates distributed memory parallelism - the foundation of HPC clusters.
Each MPI rank trains on a different subset of the data.
"""
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI
import os
from utils import (
    SimpleCNN, get_data_loaders, train_epoch, test_model,
    Timer, format_time, calculate_speedup, calculate_efficiency
)

def setup_distributed():
    """
    Initialize MPI and PyTorch distributed training
    
    Returns:
        rank, world_size, device
    """
    # Get MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    # Set up PyTorch distributed backend
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='gloo',  # Use 'gloo' for CPU (works on Mac)
        rank=rank,
        world_size=world_size
    )
    
    # Use CPU for MPI processes
    device = torch.device('cpu')
    
    return rank, world_size, device

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_distributed(epochs=5, batch_size=64, learning_rate=0.01):
    """
    Train model using MPI data parallelism
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size per process
        learning_rate: Learning rate for optimizer
    
    Returns:
        Dictionary with training results (only from rank 0)
    """
    # Setup distributed training
    rank, world_size, device = setup_distributed()
    
    # Only rank 0 prints
    is_main = (rank == 0)
    
    if is_main:
        print("\n" + "="*70)
        print(" MPI DISTRIBUTED TRAINING - DATA PARALLELISM")
        print("="*70)
        print(f"\nMPI Configuration:")
        print(f"  World Size (Processes): {world_size}")
        print(f"  Device per Process: {device}")
        print(f"  Batch Size per Process: {batch_size}")
        print(f"  Effective Batch Size: {batch_size * world_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning Rate: {learning_rate}\n")
    
    # Load data with distributed sampler
    if is_main:
        print("Loading MNIST dataset (distributed)...")
    
    train_loader, test_loader = get_data_loaders(
        batch_size=batch_size,
        is_distributed=True,
        rank=rank,
        world_size=world_size
    )
    
    if is_main:
        print(f"Training samples per process: {len(train_loader.dataset) // world_size}")
        print(f"Total training samples: {len(train_loader.dataset)}")
    
    # Create model and wrap with DDP
    if is_main:
        print("\nInitializing distributed model...")
    
    model = SimpleCNN().to(device)
    model = DDP(model)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
    
    # Training
    if is_main:
        print("\n" + "-"*70)
        print("Starting distributed training...")
        print("-"*70)
    
    timer = Timer()
    timer.start()
    
    epoch_times = []
    train_losses = []
    test_accuracies = []
    
    for epoch in range(1, epochs + 1):
        # Synchronize all processes at epoch start
        dist.barrier()
        
        epoch_timer = Timer()
        epoch_timer.start()
        
        if is_main:
            print(f"\nEpoch {epoch}/{epochs} (Rank {rank})")
        
        # Set epoch for distributed sampler (ensures different shuffle each epoch)
        train_loader.sampler.set_epoch(epoch)
        
        # Train
        avg_loss = train_epoch(model, device, train_loader, optimizer, epoch, verbose=False)
        
        # Synchronize losses across all processes
        avg_loss_tensor = torch.tensor([avg_loss]).to(device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / world_size
        
        train_losses.append(avg_loss)
        
        # Test (only rank 0 runs full test)
        if is_main:
            test_loss, accuracy = test_model(model, device, test_loader, verbose=False)
            test_accuracies.append(accuracy)
        
        # Synchronize all processes
        dist.barrier()
        
        epoch_timer.stop()
        epoch_times.append(epoch_timer.elapsed())
        
        if is_main:
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Test Accuracy: {accuracy:.2f}%")
            print(f"  Epoch Time: {epoch_timer.elapsed():.2f}s")
    
    timer.stop()
    total_time = timer.elapsed()
    
    # Results (only rank 0 returns and prints)
    if is_main:
        print("\n" + "="*70)
        print(" DISTRIBUTED TRAINING COMPLETE")
        print("="*70)
        print(f"Total Training Time: {total_time:.2f} seconds")
        print(f"Average Time per Epoch: {sum(epoch_times)/len(epoch_times):.2f}s")
        print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        
        # Load baseline for comparison
        try:
            import json
            with open('baseline_results.json', 'r') as f:
                baseline = json.load(f)
                baseline_time = baseline['total_time']
                speedup = calculate_speedup(baseline_time, total_time)
                efficiency = calculate_efficiency(speedup, world_size)
                
                print(f"\nPerformance vs Baseline:")
                print(f"  Baseline Time: {baseline_time:.2f}s")
                print(f"  MPI Time: {total_time:.2f}s")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Parallel Efficiency: {efficiency:.1f}%")
        except FileNotFoundError:
            print("\nNote: Run baseline_serial.py first to compare performance")
        
        print("="*70 + "\n")
        
        results = {
            'total_time': total_time,
            'epoch_times': epoch_times,
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1],
            'world_size': world_size,
            'device': str(device),
            'method': f'MPI Distributed ({world_size} processes)'
        }
    else:
        results = None
    
    # Cleanup
    cleanup_distributed()
    
    return results

def main():
    """Main function"""
    # Get MPI info before distributed setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    if rank == 0:
        print("="*70)
        print(" SYSTEM INFORMATION")
        print("="*70)
        print(f"MPI Processes: {world_size}")
        print(f"PyTorch Version: {torch.__version__}")
        print("="*70 + "\n")
    
    # Run distributed training
    results = train_distributed(epochs=5, batch_size=64, learning_rate=0.01)
    
    # Save results (only rank 0)
    if rank == 0 and results:
        import json
        with open('mpi_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✓ Results saved to mpi_results.json")
        print("\nNext, run: python3 gpu_accelerated.py")

if __name__ == "__main__":
    main()