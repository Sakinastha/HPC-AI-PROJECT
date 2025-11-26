# ⚡ QUICKSTART GUIDE - Get Running in 10 Minutes!

## 🎯 Your HPC Project: Ready to Go!

This is a **production-grade HPC project** showing distributed computing + GPU acceleration for AI.

---

## 📋 What You Need

- ✅ MacBook M5 (you have this!)
- ✅ 10 minutes of time
- ✅ Terminal access

---

## 🚀 SETUP (5 minutes)

### 1. Install MPI
```bash
brew install open-mpi
```

### 2. Install Python Packages
```bash
pip3 install torch torchvision mpi4py matplotlib pandas numpy
```

### 3. Verify Setup
```bash
# Check MPI
mpirun --version

# Check PyTorch MPS
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

---

## 🎬 RUN IT (5 minutes)

### Option A: Run Everything Automatically
```bash
python3 benchmark_suite.py
```

**This does EVERYTHING:**
- Runs all 4 implementations
- Compares performance
- Generates graphs
- Creates report
- **Takes ~5 minutes total**

### Option B: Run Individual Tests

```bash
# 1. Baseline (2 min)
python3 baseline_serial.py

# 2. MPI Distributed (30 sec)
mpirun -n 4 python3 mpi_distributed.py

# 3. GPU Accelerated (15 sec)
python3 gpu_accelerated.py

# 4. Hybrid MPI+GPU (10 sec)
mpirun -n 4 python3 hybrid_mpi_gpu.py
```

---

## 📊 YOUR RESULTS

After running, you'll have:

```
hpc_ai_project/
├── hpc_performance_analysis.png  ← Beautiful performance graphs
├── project_report.txt             ← Your complete analysis
├── complete_results.json          ← All data
└── (4 individual result files)
```

---

## 🎓 WHAT THIS SHOWS YOUR PROFESSOR

### ✅ HPC Techniques Demonstrated:

1. **MPI (Message Passing Interface)**
   - Distributed memory parallelism
   - The foundation of ALL HPC clusters
   - Data parallelism across processes

2. **GPU Acceleration**
   - Apple Metal Performance Shaders
   - Heterogeneous computing (CPU+GPU)
   - Modern HPC requirement

3. **Hybrid Parallelism**
   - MPI + GPU combined
   - State-of-the-art approach
   - Used by modern supercomputers

4. **Performance Analysis**
   - Speedup calculations
   - Parallel efficiency
   - Scaling studies
   - Communication overhead analysis

---

## 📈 EXPECTED RESULTS

On your M5 Mac, you should see:

| Method | Time | Speedup |
|--------|------|---------|
| Serial CPU | ~120s | 1.0x |
| MPI (4 proc) | ~35s | 3.4x |
| GPU (MPS) | ~20s | 6.0x |
| Hybrid | ~12s | 10x |

*Actual times may vary*

---

## 🔥 KEY POINTS FOR YOUR PRESENTATION

1. **"This uses MPI - the standard for HPC clusters"**
   - Same code runs on supercomputers
   - Scalable to 1000s of nodes

2. **"GPU acceleration is essential for modern AI"**
   - GPT-3 trained on 10,000+ GPUs
   - Shows heterogeneous computing

3. **"Hybrid approach combines best of both"**
   - Industry standard (Summit, Frontier)
   - Demonstrates understanding of real HPC

4. **"Applied to actual AI workload"**
   - Neural network training
   - Matrix operations are core of AI
   - Directly relevant to deep learning

---

## 🆘 TROUBLESHOOTING

### "mpirun: command not found"
```bash
brew install open-mpi
# Then restart terminal
```

### "No module named 'mpi4py'"
```bash
pip3 install mpi4py
```

### "MPS not available"
```bash
# Check macOS version (need 12.3+)
sw_vers

# Update PyTorch
pip3 install --upgrade torch
```

### MPI hangs on Mac
```bash
# Try fewer processes
mpirun -n 2 python3 mpi_distributed.py

# Or set this environment variable
export OMPI_MCA_btl=^openib
```

---

## 💡 PRO TIPS

### Speed up testing:
Edit files and change `epochs=5` to `epochs=2` (faster!)

### Test with more processes:
```bash
mpirun -n 8 python3 mpi_distributed.py
```

### Use different dataset:
Replace MNIST with CIFAR-10 in the code

---
