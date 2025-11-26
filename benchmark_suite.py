"""
Comprehensive Benchmark Suite
HPC AI Project: Automated testing and performance analysis

This script runs all implementations and generates:
1. Performance comparison plots
2. Scaling analysis
3. Detailed report
"""
import subprocess
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def run_baseline():
    """Run baseline serial training"""
    print("\n" + "="*70)
    print(" RUNNING BASELINE (SERIAL CPU)")
    print("="*70)
    result = subprocess.run(['python3', 'baseline/serial_training.py'],
                            capture_output=False, text=True)
    if result.returncode != 0:
        print("⚠️  Baseline failed to run")
        return None
    try:
        with open('baseline_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️  Baseline results not found")
        return None

def run_mpi(num_processes=4):
    """Run MPI distributed training"""
    print("\n" + "="*70)
    print(f" RUNNING MPI ({num_processes} PROCESSES)")
    print("="*70)
    result = subprocess.run(['mpirun', '-n', str(num_processes),
                             'python3', 'mpi_distributed/mpi_parallel_training.py'],
                            capture_output=False, text=True)
    if result.returncode != 0:
        print("⚠️  MPI training failed")
        return None
    try:
        with open('mpi_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️  MPI results not found")
        return None

def run_gpu():
    """Run GPU accelerated training"""
    print("\n" + "="*70)
    print(" RUNNING GPU (APPLE MPS)")
    print("="*70)
    result = subprocess.run(['python3', 'gpu_accelerated/mps_gpu_training.py'],
                           capture_output=False, text=True)
    if result.returncode != 0:
        print("⚠️  GPU training failed")
        return None
    try:
        with open('gpu_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️  GPU results not found")
        return None

def run_hybrid(num_processes=4):
    """Run hybrid MPI+GPU training"""
    print("\n" + "="*70)
    print(f" RUNNING HYBRID (MPI+GPU, {num_processes} PROCESSES)")
    print("="*70)
    result = subprocess.run(['mpirun', '-n', str(num_processes),
                             'python3', 'hybrid/mpi_gpu_hybrid.py'],
                            capture_output=False, text=True)
    if result.returncode != 0:
        print("⚠️  Hybrid training failed")
        return None
    try:
        with open('hybrid_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️  Hybrid results not found")
        return None


def create_visualizations(results):
    """Create comprehensive performance plots"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Extract data
    methods = []
    times = []
    accuracies = []
    
    if results['baseline']:
        methods.append('Serial\n(CPU)')
        times.append(results['baseline']['total_time'])
        accuracies.append(results['baseline']['final_accuracy'])
    
    if results['mpi']:
        methods.append(f"MPI\n({results['mpi']['world_size']} proc)")
        times.append(results['mpi']['total_time'])
        accuracies.append(results['mpi']['final_accuracy'])
    
    if results['gpu']:
        methods.append('GPU\n(MPS)')
        times.append(results['gpu']['total_time'])
        accuracies.append(results['gpu']['final_accuracy'])
    
    if results['hybrid']:
        methods.append(f"Hybrid\n(MPI+GPU)")
        times.append(results['hybrid']['total_time'])
        accuracies.append(results['hybrid']['final_accuracy'])
    
    # Calculate speedups
    baseline_time = results['baseline']['total_time'] if results['baseline'] else times[0]
    speedups = [baseline_time / t for t in times]
    
    # Plot 1: Execution Time Comparison
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(methods, times, color=['gray', 'blue', 'green', 'red'][:len(methods)])
    ax1.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Execution Time Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, time_val in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Speedup
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(methods, speedups, color=['gray', 'blue', 'green', 'red'][:len(methods)])
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_ylabel('Speedup vs Baseline', fontsize=11, fontweight='bold')
    ax2.set_title('Performance Speedup', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, speedup in zip(bars2, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Accuracy Comparison
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(methods, accuracies, color=['gray', 'blue', 'green', 'red'][:len(methods)])
    ax3.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Model Accuracy', fontsize=13, fontweight='bold')
    ax3.set_ylim([min(accuracies) - 1, 100])
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars3, accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Training Progress (Loss over epochs)
    ax4 = plt.subplot(2, 3, 4)
    if results['baseline']:
        ax4.plot(range(1, len(results['baseline']['train_losses'])+1),
                results['baseline']['train_losses'], 'o-', label='Serial', linewidth=2)
    if results['mpi']:
        ax4.plot(range(1, len(results['mpi']['train_losses'])+1),
                results['mpi']['train_losses'], 's-', label='MPI', linewidth=2)
    if results['gpu']:
        ax4.plot(range(1, len(results['gpu']['train_losses'])+1),
                results['gpu']['train_losses'], '^-', label='GPU', linewidth=2)
    if results['hybrid']:
        ax4.plot(range(1, len(results['hybrid']['train_losses'])+1),
                results['hybrid']['train_losses'], 'd-', label='Hybrid', linewidth=2)
    
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
    ax4.set_title('Training Convergence', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Plot 5: Time per Epoch
    ax5 = plt.subplot(2, 3, 5)
    epoch_data = []
    epoch_labels = []
    
    if results['baseline']:
        epoch_data.append(np.mean(results['baseline']['epoch_times']))
        epoch_labels.append('Serial')
    if results['mpi']:
        epoch_data.append(np.mean(results['mpi']['epoch_times']))
        epoch_labels.append('MPI')
    if results['gpu']:
        epoch_data.append(np.mean(results['gpu']['epoch_times']))
        epoch_labels.append('GPU')
    if results['hybrid']:
        epoch_data.append(np.mean(results['hybrid']['epoch_times']))
        epoch_labels.append('Hybrid')
    
    bars5 = ax5.bar(epoch_labels, epoch_data, 
                   color=['gray', 'blue', 'green', 'red'][:len(epoch_data)])
    ax5.set_ylabel('Time per Epoch (seconds)', fontsize=11, fontweight='bold')
    ax5.set_title('Average Epoch Time', fontsize=13, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    for bar, time_val in zip(bars5, epoch_data):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Efficiency Analysis
    ax6 = plt.subplot(2, 3, 6)
    
    efficiency_data = []
    efficiency_labels = []
    
    if results['mpi']:
        num_proc = results['mpi']['world_size']
        mpi_speedup = baseline_time / results['mpi']['total_time']
        mpi_efficiency = (mpi_speedup / num_proc) * 100
        efficiency_data.append(mpi_efficiency)
        efficiency_labels.append(f'MPI\n({num_proc})')
    
    if results['hybrid']:
        num_proc = results['hybrid']['world_size']
        hybrid_speedup = baseline_time / results['hybrid']['total_time']
        hybrid_efficiency = (hybrid_speedup / num_proc) * 100
        efficiency_data.append(hybrid_efficiency)
        efficiency_labels.append(f'Hybrid\n({num_proc})')
    
    if efficiency_data:
        bars6 = ax6.bar(efficiency_labels, efficiency_data, 
                       color=['blue', 'red'][:len(efficiency_data)])
        ax6.axhline(y=100, color='black', linestyle='--', linewidth=2, 
                   label='Ideal (100%)', alpha=0.5)
        ax6.set_ylabel('Parallel Efficiency (%)', fontsize=11, fontweight='bold')
        ax6.set_title('Parallel Efficiency', fontsize=13, fontweight='bold')
        ax6.set_ylim([0, 120])
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        
        for bar, eff in zip(bars6, efficiency_data):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{eff:.1f}%',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('HPC AI Project: Comprehensive Performance Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('hpc_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Performance plots saved to: hpc_performance_analysis.png")
    
    return fig

def generate_report(results):
    """Generate comprehensive text report"""
    
    report = []
    report.append("="*80)
    report.append(" HPC AI PROJECT: COMPREHENSIVE PERFORMANCE REPORT")
    report.append("="*80)
    report.append("\nProject: Hybrid Parallelism for Deep Learning on Apple Silicon")
    report.append("Task: MNIST Classification using CNN")
    report.append("Parallelization Techniques: MPI, GPU (MPS), Hybrid")
    
    report.append("\n" + "="*80)
    report.append(" EXECUTION TIME RESULTS")
    report.append("="*80)
    
    baseline_time = 0
    if results['baseline']:
        baseline_time = results['baseline']['total_time']
        report.append(f"\n1. Serial (CPU Baseline)")
        report.append(f"   Total Time: {baseline_time:.2f} seconds")
        report.append(f"   Test Accuracy: {results['baseline']['final_accuracy']:.2f}%")
        report.append(f"   Avg Epoch Time: {np.mean(results['baseline']['epoch_times']):.2f}s")
    
    if results['mpi']:
        mpi_time = results['mpi']['total_time']
        speedup = baseline_time / mpi_time if baseline_time > 0 else 0
        efficiency = (speedup / results['mpi']['world_size']) * 100
        report.append(f"\n2. MPI Distributed ({results['mpi']['world_size']} processes)")
        report.append(f"   Total Time: {mpi_time:.2f} seconds")
        report.append(f"   Speedup: {speedup:.2f}x")
        report.append(f"   Parallel Efficiency: {efficiency:.1f}%")
        report.append(f"   Test Accuracy: {results['mpi']['final_accuracy']:.2f}%")
    
    if results['gpu']:
        gpu_time = results['gpu']['total_time']
        speedup = baseline_time / gpu_time if baseline_time > 0 else 0
        report.append(f"\n3. GPU Accelerated (Apple MPS)")
        report.append(f"   Total Time: {gpu_time:.2f} seconds")
        report.append(f"   Speedup: {speedup:.2f}x")
        report.append(f"   Test Accuracy: {results['gpu']['final_accuracy']:.2f}%")
    
    if results['hybrid']:
        hybrid_time = results['hybrid']['total_time']
        speedup = baseline_time / hybrid_time if baseline_time > 0 else 0
        efficiency = (speedup / results['hybrid']['world_size']) * 100
        report.append(f"\n4. Hybrid MPI+GPU ({results['hybrid']['world_size']} processes)")
        report.append(f"   Total Time: {hybrid_time:.2f} seconds")
        report.append(f"   Speedup: {speedup:.2f}x")
        report.append(f"   Parallel Efficiency: {efficiency:.1f}%")
        report.append(f"   Test Accuracy: {results['hybrid']['final_accuracy']:.2f}%")
        
        if 'communication_times' in results['hybrid']:
            comm_overhead = (sum(results['hybrid']['communication_times']) / hybrid_time) * 100
            report.append(f"   Communication Overhead: {comm_overhead:.1f}%")
    
    report.append("\n" + "="*80)
    report.append(" PERFORMANCE ANALYSIS")
    report.append("="*80)
    
    report.append("\n1. SPEEDUP ANALYSIS:")
    if baseline_time > 0:
        if results['mpi']:
            report.append(f"   • MPI: {baseline_time/results['mpi']['total_time']:.2f}x faster than serial")
        if results['gpu']:
            report.append(f"   • GPU: {baseline_time/results['gpu']['total_time']:.2f}x faster than serial")
        if results['hybrid']:
            report.append(f"   • Hybrid: {baseline_time/results['hybrid']['total_time']:.2f}x faster than serial ⭐")
    
    report.append("\n2. PARALLEL EFFICIENCY:")
    report.append("   Parallel efficiency measures how well we use additional processors.")
    report.append("   Ideal efficiency = 100% (perfect linear scaling)")
    if results['mpi']:
        num_proc = results['mpi']['world_size']
        speedup = baseline_time / results['mpi']['total_time']
        eff = (speedup / num_proc) * 100
        report.append(f"   • MPI: {eff:.1f}% (with {num_proc} processes)")
    if results['hybrid']:
        num_proc = results['hybrid']['world_size']
        speedup = baseline_time / results['hybrid']['total_time']
        eff = (speedup / num_proc) * 100
        report.append(f"   • Hybrid: {eff:.1f}% (with {num_proc} processes)")
    
    report.append("\n3. WHY EFFICIENCY < 100%:")
    report.append("   • Communication overhead between MPI processes")
    report.append("   • Load imbalance (uneven work distribution)")
    report.append("   • Serial portions of code (Amdahl's Law)")
    report.append("   • Synchronization barriers")
    
    report.append("\n" + "="*80)
    report.append(" HPC CONCEPTS DEMONSTRATED")
    report.append("="*80)
    
    report.append("\n1. DISTRIBUTED MEMORY PARALLELISM (MPI)")
    report.append("   • Message Passing Interface - standard for HPC clusters")
    report.append("   • Data parallelism: Split dataset across processes")
    report.append("   • Gradient synchronization via all-reduce")
    report.append("   • Scalable to thousands of nodes")
    
    report.append("\n2. ACCELERATOR COMPUTING (GPU)")
    report.append("   • Heterogeneous computing: CPU + GPU")
    report.append("   • Massive parallelism: Thousands of GPU cores")
    report.append("   • Apple Metal Performance Shaders (MPS)")
    report.append("   • Unified memory architecture on M5")
    
    report.append("\n3. HYBRID PARALLELISM (MPI + GPU)")
    report.append("   • State-of-the-art HPC: Multi-node with GPUs")
    report.append("   • Used by modern supercomputers (Summit, Frontier)")
    report.append("   • Best of both: Distributed scale + GPU speed")
    report.append("   • Essential for training large AI models")
    
    report.append("\n4. PERFORMANCE METRICS")
    report.append("   • Speedup: How much faster than baseline")
    report.append("   • Efficiency: How well resources are utilized")
    report.append("   • Strong Scaling: Fixed problem, more resources")
    report.append("   • Communication Overhead: Cost of coordination")
    
    report.append("\n" + "="*80)
    report.append(" RELEVANCE TO AI/DEEP LEARNING")
    report.append("="*80)
    
    report.append("\n• Neural networks are fundamentally matrix operations")
    report.append("• Training large models requires distributed computing")
    report.append("• GPT-3: Trained on 10,000+ GPUs")
    report.append("• Modern AI research depends on HPC infrastructure")
    report.append("• This project demonstrates scalable techniques")
    
    report.append("\n" + "="*80)
    report.append(" CONCLUSION")
    report.append("="*80)
    
    if results['hybrid'] and baseline_time > 0:
        total_speedup = baseline_time / results['hybrid']['total_time']
        time_saved = baseline_time - results['hybrid']['total_time']
        report.append(f"\nBest Performance: Hybrid MPI+GPU")
        report.append(f"  • {total_speedup:.2f}x faster than serial implementation")
        report.append(f"  • Saved {time_saved:.2f} seconds ({(time_saved/baseline_time)*100:.1f}%)")
        report.append(f"  • Demonstrates production HPC techniques")
    
    report.append("\nThis project successfully demonstrates:")
    report.append("✓ Distributed memory parallelism with MPI")
    report.append("✓ GPU acceleration with Apple Silicon")
    report.append("✓ Hybrid parallelism (industry standard)")
    report.append("✓ Performance analysis and optimization")
    report.append("✓ Application to real AI/ML workloads")
    
    report.append("\n" + "="*80)
    
    report_text = "\n".join(report)
    
    # Save report
    with open('project_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print("\n✓ Full report saved to: project_report.txt")

def main():
    """Main benchmark suite"""
    
    print("="*80)
    print(" HPC AI PROJECT - COMPREHENSIVE BENCHMARK SUITE")
    print("="*80)
    print("\nThis will run all implementations and generate:")
    print("  1. Performance comparison plots")
    print("  2. Scaling analysis")
    print("  3. Comprehensive report")
    print("\nEstimated time: 3-5 minutes")
    print("="*80)
    
    input("\nPress Enter to start benchmark...")
    
    start_time = time.time()
    
    # Run all benchmarks
    results = {
        'baseline': None,
        'mpi': None,
        'gpu': None,
        'hybrid': None
    }
    
    results['baseline'] = run_baseline()
    results['mpi'] = run_mpi(num_processes=4)
    results['gpu'] = run_gpu()
    results['hybrid'] = run_hybrid(num_processes=4)
    
    # Check if we have at least baseline
    if not results['baseline']:
        print("\n❌ Error: Baseline results not available. Cannot continue.")
        return
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(" GENERATING ANALYSIS")
    print("="*80)
    
    # Create visualizations
    create_visualizations(results)
    
    # Generate report
    generate_report(results)
    
    # Save all results
    with open('complete_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(" BENCHMARK COMPLETE!")
    print("="*80)
    print(f"\nTotal benchmark time: {total_time:.1f} seconds")
    print("\nGenerated files:")
    print("  1. hpc_performance_analysis.png - Performance visualizations")
    print("  2. project_report.txt - Detailed analysis report")
    print("  3. complete_results.json - All raw data")
    print("\nYour HPC project is complete! Use these files for your submission.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()