import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import print_banner  # or other functions you need


"""
Baseline Serial Training - Single CPU
HPC AI Project: Neural Network Training without parallelization

This serves as the baseline for comparison with parallel implementations.
"""
import torch
import torch.optim as optim
from utils import (
    SimpleCNN, get_data_loaders, train_epoch, test_model,
    print_system_info, Timer, print_performance_summary
)

def train_serial(epochs=5, batch_size=64, learning_rate=0.01):
    """
    Train model on single CPU (no parallelization)
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    
    Returns:
        Dictionary with training results
    """
    print("\n" + "="*70)
    print(" BASELINE: SERIAL TRAINING (SINGLE CPU)")
    print("="*70)
    
    # Force CPU usage (no GPU, no parallelization)
    device = torch.device('cpu')
    print(f"\nDevice: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}\n")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training
    print("\n" + "-"*70)
    print("Starting training...")
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
    print(" TRAINING COMPLETE")
    print("="*70)
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Average Time per Epoch: {sum(epoch_times)/len(epoch_times):.2f}s")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print("="*70 + "\n")
    
    return {
        'total_time': total_time,
        'epoch_times': epoch_times,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'final_accuracy': test_accuracies[-1],
        'device': str(device),
        'method': 'Serial (CPU)'
    }

def main():
    """Main function"""
    print_system_info()
    
    # Run serial training
    results = train_serial(epochs=5, batch_size=64, learning_rate=0.01)
    
    # Save results
    import json
    with open('baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Results saved to baseline_results.json")
    print("\nThis baseline will be used to calculate speedup for parallel implementations.")
    print("Next, run: mpirun -n 4 python3 mpi_distributed.py")

if __name__ == "__main__":
    main()