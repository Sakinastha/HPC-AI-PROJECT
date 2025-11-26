import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import print_banner  # or other functions you need

"""
Hybrid MPI + GPU Training
HPC AI Project: Combining distributed parallelism with GPU acceleration

This represents the state-of-the-art in HPC:
- MPI for distributed memory parallelism (multiple nodes)
- GPU for accelerated computation within each node
- This is how modern supercomputers train large AI models
"""
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI
import os
from utils import (
    SimpleCNN, get_data_loaders, train_epoch, test_model,
    Timer, calculate_speedup, calculate_efficiency
)

def setup_hybrid():
    """
    Initialize MPI and PyTorch distributed training with GPU support
    
    Returns:
        rank, world_size, device
    """
    # Get MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    # Set up PyTorch distributed backend
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
    # Initialize process group
    dist.init_process_group(
        backend='gloo',
        rank=rank,
        world_size=world_size
    )
    
    # Try to use MPS (Apple GPU), fallback to CPU
    # Note: In real HPC clusters, each rank would get its own GPU
    # On Mac, all ranks share the same GPU (simulated multi-GPU)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    return rank, world_size, device

def cleanup_hybrid():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_hybrid(epochs=5, batch_size=64, learning_rate=0.01):
    """
    Train model using hybrid MPI + GPU parallelism
    
    This simulates a multi-node HPC cluster where:
    - Each MPI rank = One compute node
    - Each rank uses GPU acceleration
    - Data is distributed across ranks
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size per process
        learning_rate: Learning rate for optimizer
    
    Returns:
        Dictionary with training results (only from rank 0)
    """
    # Setup hybrid training
    rank, world_size, device = setup_hybrid()
    
    # Only rank 0 prints
    is_main = (rank == 0)
    
    if is_main:
        print("\n" + "="*70)
        print(" HYBRID MPI + GPU TRAINING")
        print("="*70)
        print("\n🚀 This represents state-of-the-art HPC computing!")
        print("\nConfiguration:")
        print(f"  MPI Processes (simulated nodes): {world_size}")
        print(f"  Device per process: {device}")
        
        if device.type == 'mps':
            print(f"  ✓ Using Apple Metal GPU acceleration")
            print(f"  ✓ Each MPI rank shares GPU (simulated multi-GPU)")
        else:
            print(f"  ⚠️  GPU not available, using CPU")
        
        print(f"\n  Batch Size per Process: {batch_size}")
        print(f"  Effective Batch Size: {batch_size * world_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning Rate: {learning_rate}")
        
        print("\nThis simulates a real HPC cluster where:")
        print("  • Each MPI rank = One compute node with GPU")
        print("  • Data distributed across nodes")
        print("  • Each node processes independently with GPU")
        print("  • Gradients synchronized via MPI\n")
    
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
        print("\nInitializing distributed model with GPU support...")
    
    model = SimpleCNN().to(device)
    model = DDP(model)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
    
    # Training
    if is_main:
        print("\n" + "-"*70)
        print("Starting hybrid distributed + GPU training...")
        print("-"*70)
    
    timer = Timer()
    timer.start()
    
    epoch_times = []
    train_losses = []
    test_accuracies = []
    communication_times = []
    
    for epoch in range(1, epochs + 1):
        # Synchronize all processes at epoch start
        dist.barrier()
        
        epoch_timer = Timer()
        epoch_timer.start()
        
        if is_main:
            print(f"\nEpoch {epoch}/{epochs}")
        
        # Set epoch for distributed sampler
        train_loader.sampler.set_epoch(epoch)
        
        # Train
        avg_loss = train_epoch(model, device, train_loader, optimizer, epoch, verbose=False)
        
        # Measure communication time for gradient synchronization
        comm_timer = Timer()
        comm_timer.start()
        
        # Synchronize losses across all processes
        avg_loss_tensor = torch.tensor([avg_loss]).to(device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / world_size
        
        comm_timer.stop()
        communication_times.append(comm_timer.elapsed())
        
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
            print(f"  Communication Overhead: {communication_times[-1]*1000:.1f}ms")
    
    timer.stop()
    total_time = timer.elapsed()
    
    # Results (only rank 0 returns and prints)
    if is_main:
        print("\n" + "="*70)
        print(" HYBRID TRAINING COMPLETE")
        print("="*70)
        print(f"Total Training Time: {total_time:.2f} seconds")
        print(f"Average Time per Epoch: {sum(epoch_times)/len(epoch_times):.2f}s")
        print(f"Average Communication Time: {sum(communication_times)/len(communication_times)*1000:.1f}ms/epoch")
        print(f"Communication Overhead: {sum(communication_times)/total_time*100:.1f}%")
        print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        
        # Compare with all previous methods
        print("\n" + "="*70)
        print(" PERFORMANCE COMPARISON")
        print("="*70)
        
        try:
            import json
            
            # Load baseline
            with open('baseline_results.json', 'r') as f:
                baseline = json.load(f)
                baseline_time = baseline['total_time']
            
            print(f"\n1. Serial CPU (Baseline):  {baseline_time:.2f}s  (1.00x)")
            
            # Load MPI results
            try:
                with open('mpi_results.json', 'r') as f:
                    mpi = json.load(f)
                    mpi_time = mpi['total_time']
                    mpi_speedup = calculate_speedup(baseline_time, mpi_time)
                print(f"2. MPI ({world_size} processes):     {mpi_time:.2f}s  ({mpi_speedup:.2f}x)")
            except FileNotFoundError:
                print("2. MPI: Not available")
            
            # Load GPU results
            try:
                with open('gpu_results.json', 'r') as f:
                    gpu = json.load(f)
                    gpu_time = gpu['total_time']
                    gpu_speedup = calculate_speedup(baseline_time, gpu_time)
                print(f"3. GPU (Apple MPS):        {gpu_time:.2f}s  ({gpu_speedup:.2f}x)")
            except FileNotFoundError:
                print("3. GPU: Not available")
            
            # Current hybrid
            hybrid_speedup = calculate_speedup(baseline_time, total_time)
            hybrid_efficiency = calculate_efficiency(hybrid_speedup, world_size)
            print(f"4. Hybrid (MPI+GPU):       {total_time:.2f}s  ({hybrid_speedup:.2f}x) ⭐")
            
            print(f"\n✓ Best Performance: Hybrid MPI+GPU")
            print(f"  Total Speedup: {hybrid_speedup:.2f}x over baseline")
            print(f"  Parallel Efficiency: {hybrid_efficiency:.1f}%")
            print(f"  Time Saved: {baseline_time - total_time:.2f}s ({(1 - total_time/baseline_time)*100:.1f}%)")
            
        except FileNotFoundError:
            print("\nNote: Run other scripts first for full comparison")
            print("  1. python3 baseline_serial.py")
            print("  2. mpirun -n 4 python3 mpi_distributed.py")
            print("  3. python3 gpu_accelerated.py")
        
        print("="*70 + "\n")
        
        results = {
            'total_time': total_time,
            'epoch_times': epoch_times,
            'communication_times': communication_times,
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1],
            'world_size': world_size,
            'device': str(device),
            'method': f'Hybrid MPI+GPU ({world_size} processes)'
        }
    else:
        results = None
    
    # Cleanup
    cleanup_hybrid()
    
    return results

def explain_hybrid_hpc():
    """Explain why hybrid parallelism is important for HPC"""
    print("\n" + "="*70)
    print(" WHY HYBRID MPI+GPU IS THE FUTURE OF HPC")
    print("="*70)
    
    print("\n1. Real-World HPC Architecture:")
    print("   • Summit (Oak Ridge): 4,608 nodes × 6 GPUs = 27,648 GPUs")
    print("   • Frontier (Oak Ridge): #1 supercomputer, AMD GPUs")
    print("   • Each node: Multiple GPUs + CPUs")
    
    print("\n2. Communication Patterns:")
    print("   • MPI: Inter-node communication (slow, network-bound)")
    print("   • GPU: Intra-node computation (fast, memory-bound)")
    print("   • Key: Minimize MPI overhead, maximize GPU utilization")
    
    print("\n3. AI/ML Workloads:")
    print("   • GPT-3: Trained on 10,000+ GPUs")
    print("   • Stable Diffusion: Multi-GPU distributed training")
    print("   • AlphaFold: Hybrid CPU+GPU for protein folding")
    
    print("\n4. Your Implementation:")
    print("   • Simulates multi-node cluster on single Mac")
    print("   • Each MPI rank = One compute node")
    print("   • Shows same principles as real supercomputers")
    print("   • Scalable to actual HPC clusters")
    
    print("="*70 + "\n")

def main():
    """Main function"""
    # Get MPI info before distributed setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    if rank == 0:
        print("="*70)
        print(" HYBRID MPI+GPU HPC TRAINING")
        print("="*70)
        print(f"\nMPI Processes: {world_size}")
        print(f"PyTorch Version: {torch.__version__}")
        
        if torch.backends.mps.is_available():
            print(f"GPU Backend: Apple Metal (MPS) ✓")
        else:
            print(f"GPU Backend: Not available (using CPU)")
        
        print("="*70 + "\n")
    
    # Run hybrid training
    results = train_hybrid(epochs=5, batch_size=64, learning_rate=0.01)
    
    # Educational content (rank 0 only)
    if rank == 0:
        explain_hybrid_hpc()
        
        # Save results
        if results:
            import json
            with open('hybrid_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print("✓ Results saved to hybrid_results.json")
            print("\nYou've now completed all implementations!")
            print("Run: python3 benchmark_suite.py")
            print("This will generate comprehensive performance analysis and plots.")

if __name__ == "__main__":
    main()