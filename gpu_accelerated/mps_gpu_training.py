import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import print_banner  # or other functions you need

"""
GPU Accelerated Training - Apple Metal Performance Shaders
HPC AI Project: Single-GPU training using Apple Silicon's Metal backend

This demonstrates heterogeneous computing (CPU + GPU) which is
essential for modern HPC systems.
"""
import torch
import torch.optim as optim
from utils import (
    SimpleCNN, get_data_loaders, train_epoch, test_model,
    print_system_info, Timer, calculate_speedup
)

def train_gpu(epochs=5, batch_size=64, learning_rate=0.01):
    """
    Train model on Apple Silicon GPU using MPS backend
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    
    Returns:
        Dictionary with training results
    """
    print("\n" + "="*70)
    print(" GPU ACCELERATED TRAINING - APPLE METAL (MPS)")
    print("="*70)
    
    # Check for MPS availability
    if not torch.backends.mps.is_available():
        print("\n⚠️  MPS (Metal Performance Shaders) not available!")
        print("Using CPU instead. Note: This should work on M5 Macs with macOS 12.3+")
        device = torch.device('cpu')
    else:
        if not torch.backends.mps.is_built():
            print("\n⚠️  MPS not properly built in PyTorch.")
            print("Using CPU. Try: pip install --upgrade torch torchvision")
            device = torch.device('cpu')
        else:
            device = torch.device('mps')
            print("\n✓ Using Apple Metal Performance Shaders (GPU)")
    
    print(f"\nDevice: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}\n")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model and move to GPU
    print(f"\nInitializing model on {device}...")
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    if device.type == 'mps':
        print("\nGPU Memory Info:")
        print(f"  Apple M5 Unified Memory Architecture")
        print(f"  Shared CPU/GPU memory pool")
        print(f"  No explicit memory management needed")
    
    # Training
    print("\n" + "-"*70)
    print("Starting GPU training...")
    print("-"*70)
    
    timer = Timer()
    timer.start()
    
    epoch_times = []
    train_losses = []
    test_accuracies = []
    
    for epoch in range(1, epochs + 1):
        epoch_timer = Timer()
        epoch_timer.start()
        
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        avg_loss = train_epoch(model, device, train_loader, optimizer, epoch, verbose=False)
        train_losses.append(avg_loss)
        
        # Test
        test_loss, accuracy = test_model(model, device, test_loader, verbose=False)
        test_accuracies.append(accuracy)
        
        epoch_timer.stop()
        epoch_times.append(epoch_timer.elapsed())
        
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.2f}%")
        print(f"  Epoch Time: {epoch_timer.elapsed():.2f}s")
    
    timer.stop()
    total_time = timer.elapsed()
    
    # Results
    print("\n" + "="*70)
    print(" GPU TRAINING COMPLETE")
    print("="*70)
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Average Time per Epoch: {sum(epoch_times)/len(epoch_times):.2f}s")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    
    # Compare with baseline
    try:
        import json
        with open('baseline_results.json', 'r') as f:
            baseline = json.load(f)
            baseline_time = baseline['total_time']
            speedup = calculate_speedup(baseline_time, total_time)
            
            print(f"\nPerformance vs Baseline (CPU):")
            print(f"  CPU Time: {baseline_time:.2f}s")
            print(f"  GPU Time: {total_time:.2f}s")
            print(f"  GPU Speedup: {speedup:.2f}x")
            print(f"  Time Saved: {baseline_time - total_time:.2f}s ({(1 - total_time/baseline_time)*100:.1f}%)")
    except FileNotFoundError:
        print("\nNote: Run baseline_serial.py first to compare performance")
    
    print("="*70 + "\n")
    
    return {
        'total_time': total_time,
        'epoch_times': epoch_times,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'final_accuracy': test_accuracies[-1],
        'device': str(device),
        'method': 'GPU (Apple MPS)'
    }

def demonstrate_gpu_advantages():
    """
    Demonstrate why GPU acceleration matters for AI/HPC
    """
    print("\n" + "="*70)
    print(" WHY GPU ACCELERATION MATTERS FOR AI")
    print("="*70)
    
    print("\n1. Parallel Architecture:")
    print("   - Apple M5 GPU: Thousands of parallel cores")
    print("   - Perfect for matrix operations (the heart of neural networks)")
    print("   - Can process entire batches simultaneously")
    
    print("\n2. Unified Memory Architecture (M5 Advantage):")
    print("   - CPU and GPU share same memory pool")
    print("   - No PCIe bottleneck (unlike discrete GPUs)")
    print("   - Zero-copy data transfer")
    
    print("\n3. Real-World Impact:")
    print("   - Training large models (GPT, ResNet): Hours → Minutes")
    print("   - Research iteration speed: 10x faster experimentation")
    print("   - Production inference: Real-time processing possible")
    
    print("\n4. HPC Relevance:")
    print("   - Modern supercomputers: 1000s of GPUs")
    print("   - Hybrid computing: CPU for control, GPU for computation")
    print("   - Example: Summit (Oak Ridge): 27,648 NVIDIA GPUs")
    
    print("="*70 + "\n")

def main():
    """Main function"""
    print_system_info()
    
    # Check MPS availability explicitly
    if torch.backends.mps.is_available():
        print("✓ Metal Performance Shaders (MPS) is available!")
        print("  Your M5 Mac GPU will be used for acceleration.\n")
    else:
        print("⚠️  MPS not available. Training will use CPU.")
        print("  For GPU acceleration, ensure:")
        print("    - macOS 12.3 or later")
        print("    - PyTorch 1.12 or later")
        print("    - Run: pip install --upgrade torch torchvision\n")
    
    # Run GPU training
    results = train_gpu(epochs=5, batch_size=64, learning_rate=0.01)
    
    # Educational content
    demonstrate_gpu_advantages()
    
    # Save results
    import json
    with open('gpu_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Results saved to gpu_results.json")
    print("\nNext, run: mpirun -n 4 python3 hybrid_mpi_gpu.py")
    print("(This combines MPI distributed training WITH GPU acceleration)")

if __name__ == "__main__":
    main()