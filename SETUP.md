# HPC Project Setup Guide - Apple M5 MacBook

## 🎯 Project: Hybrid Parallelism for Deep Learning on Apple Silicon

This guide will get you from zero to running distributed HPC training in 15 minutes.

---

## Step 1: Install Homebrew (if not installed)

```bash
# Check if you have Homebrew
which brew

# If not installed, run:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

---

## Step 2: Install Open MPI

```bash
# Install Open MPI for distributed computing
brew install open-mpi

# Verify installation
mpirun --version
# Should show: mpirun (Open MPI) 5.x.x or similar
```

---

## Step 3: Install Python Dependencies

```bash
# Create project directory
mkdir hpc_ai_project
cd hpc_ai_project

# Install required packages
pip3 install torch torchvision torchaudio
pip3 install mpi4py
pip3 install matplotlib pandas numpy
pip3 install scikit-learn

# Verify PyTorch has MPS support
python3 -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
# Should print: MPS Available: True
```

**Important for M5 Mac:**
- PyTorch automatically supports Metal Performance Shaders (MPS)
- MPS is Apple's GPU acceleration framework
- Your M5 chip has powerful GPU cores that MPS will utilize

---

## Step 4: Verify MPI Setup

```bash
# Test MPI with 4 processes
mpirun -n 4 python3 -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()} of {MPI.COMM_WORLD.Get_size()}')"

# Should see output from 4 different processes:
# Rank 0 of 4
# Rank 1 of 4
# Rank 2 of 4
# Rank 3 of 4
```

---

## Step 5: Download Project Files

Copy all the Python files I've created into your `hpc_ai_project` directory:

1. `baseline_serial.py`
2. `mpi_distributed.py`
3. `gpu_accelerated.py`
4. `hybrid_mpi_gpu.py`
5. `benchmark_suite.py`
6. `utils.py`

---

## Step 6: Quick Test

```bash
# Test serial baseline (should take ~2 min)
python3 baseline_serial.py

# Test MPI (should see 4 processes)
mpirun -n 4 python3 mpi_distributed.py

# Test GPU (should see MPS device)
python3 gpu_accelerated.py
```

---

## Step 7: Run Full Benchmark

```bash
# This runs everything and generates results
python3 benchmark_suite.py
```

Expected output files:
- `hpc_scaling_results.png` - Performance plots
- `performance_report.txt` - Detailed analysis
- `results.json` - Raw data

---

## 🔧 Troubleshooting

### Issue: "No module named 'mpi4py'"
```bash
pip3 install mpi4py
```

### Issue: "MPS not available"
- Make sure you're on macOS 12.3 or later
- Update PyTorch: `pip3 install --upgrade torch`

### Issue: "mpirun command not found"
```bash
brew install open-mpi
# Then restart terminal
```

### Issue: MPI processes hang
- This can happen on some Mac configurations
- Try: `export OMPI_MCA_btl=^openib`
- Or use fewer processes: `mpirun -n 2` instead of `-n 4`

---

## 📊 What Each File Does

| File | Purpose | Runtime |
|------|---------|---------|
| `baseline_serial.py` | Single CPU training (baseline) | ~2 min |
| `mpi_distributed.py` | MPI data parallelism | ~30 sec |
| `gpu_accelerated.py` | Apple Metal GPU training | ~15 sec |
| `hybrid_mpi_gpu.py` | MPI + GPU combined | ~10 sec |
| `benchmark_suite.py` | Run all & generate report | ~5 min |

---

## 🎯 Success Checklist

- [ ] MPI installed and working (`mpirun --version`)
- [ ] Python packages installed (`import torch, mpi4py`)
- [ ] MPS available (`torch.backends.mps.is_available() == True`)
- [ ] Can run 4 MPI processes
- [ ] Baseline script runs successfully
- [ ] Benchmark suite completes

---

## 🚀 Ready to Run!

Once all checks pass, you're ready to run your complete HPC project:

```bash
python3 benchmark_suite.py
```

This will automatically:
1. Train models with different parallelization strategies
2. Measure execution time and speedup
3. Generate performance plots
4. Create a detailed report

**Total runtime: ~5 minutes**

---

## 💡 Pro Tips

1. **For faster testing:** Reduce epochs in each file (change `epochs=5` to `epochs=1`)
2. **For more processes:** Try `mpirun -n 8` (uses more CPU cores)
3. **For larger models:** Edit the model architecture in the files
4. **For different datasets:** Replace MNIST with CIFAR-10

---

## 📚 Understanding the Output

After running benchmark_suite.py, your report will show:

- **Baseline time:** How long serial training takes
- **MPI speedup:** How much faster with distributed training
- **GPU speedup:** How much faster with Apple Metal
- **Hybrid speedup:** Combined MPI + GPU acceleration
- **Efficiency:** How well parallelism scales
- **Communication overhead:** Cost of MPI message passing

---

