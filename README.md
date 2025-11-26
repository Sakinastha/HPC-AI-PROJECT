# HPC Project: AI-Accelerating Computing with Parallel Matrix Operations

**Course:** High Performance Computing (HPC)  
**Topic:** Accelerating AI Matrix Multiplication using HPC Techniques  
**Platform:** MacBook M5 (2025)

---

## 🎯 Project Overview

This project demonstrates how High Performance Computing techniques can accelerate the fundamental operations in AI and deep learning. Matrix multiplication is the core computational operation in neural networks, and optimizing it has direct impact on training and inference speed.

### Key Implementations:
1. **Serial Baseline** - Standard NumPy implementation
2. **Parallel Processing** - Multiprocessing with row-block distribution
3. **Cache Optimization** - Blocked matrix multiplication with Numba JIT

---

## 🚀 Quick Start (3-Step Setup)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Complete Benchmark
```bash
python benchmark.py
```

This will:
- Test multiple matrix sizes (256x256 to 2048x2048)
- Compare serial, parallel, and optimized implementations
- Generate performance graphs automatically
- Create a detailed report

### Step 3: View Results
After running, you'll have:
- `hpc_performance_results.png` - Visualization graphs
- `project_report.txt` - Detailed analysis
- `benchmark_results.json` - Raw performance data

---

## 📁 Project Structure

```
hpc_matrix_project/
├── matrix_multiply_serial.py      # Baseline implementations
├── matrix_multiply_parallel.py    # Multiprocessing version
├── matrix_multiply_optimized.py   # Cache-blocked + Numba
├── benchmark.py                    # Main benchmarking suite
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🔬 Technical Details

### 1. Serial Implementation (Baseline)
- Uses NumPy's `np.dot()` which leverages Apple's Accelerate framework on M5
- Provides baseline for speedup calculations
- Time complexity: O(n³)

### 2. Parallel Implementation
**Strategy:** Row-block distribution across processes
- Divides matrix A into horizontal blocks
- Each process computes its block independently
- Uses Python's `multiprocessing` module
- Tests with 1, 2, 4, and all available cores

**Key Code:**
```python
def matrix_multiply_parallel(A, B, num_processes):
    # Split A into row blocks
    # Distribute to worker processes
    # Assemble final result
```

### 3. Optimized Implementation
**Techniques:**
- **Cache Blocking:** Processes data in blocks that fit in cache
- **Numba JIT:** Just-in-time compilation to machine code
- **Parallel Loops:** Uses Numba's `prange` for automatic parallelization

**Why Cache Blocking?**
- Improves cache hit rate
- Reduces memory bandwidth requirements
- Better utilizes CPU's memory hierarchy

---

## 📊 Expected Results

On an M5 MacBook, you should see:

### Speedup Examples (1024x1024 matrix):
- **2 processes:** ~1.5-1.8x speedup
- **4 processes:** ~2.5-3.0x speedup
- **All cores:** ~3.5-5.0x speedup
- **Cache-blocked:** ~1.5-2.5x speedup (plus scales with parallelization)

### Efficiency:
- Efficiency typically 60-80% (varies with matrix size)
- Larger matrices show better efficiency
- Communication overhead affects smaller matrices

---

## 🤖 Relevance to AI

### Why This Matters:
1. **Neural Networks:** Every layer performs matrix multiplication
2. **Training:** Forward and backward passes are matrix operations
3. **Inference:** Real-time predictions require fast matrix ops
4. **Scaling:** Large models (GPT, BERT) require billions of operations

### Example Application:
A simple neural network forward pass:
```
Input (784) × Weights1 (784×512) → Hidden (512)
Hidden (512) × Weights2 (512×10) → Output (10)
```

For a batch of 128 images, this is ~100M operations. HPC techniques make this feasible in real-time.

---

## 🛠️ Troubleshooting

### If you get "numba not found":
```bash
pip install numba
```

### If plots don't show:
The script auto-saves to PNG. Check `hpc_performance_results.png`

### For Apple M5 specific:
- No CUDA/OpenMP needed (not supported on Mac anyway)
- NumPy automatically uses Apple's Accelerate framework
- Multiprocessing works perfectly on Apple Silicon

---

## 📈 Running Individual Tests

### Test only serial:
```bash
python matrix_multiply_serial.py
```

### Test only parallel:
```bash
python matrix_multiply_parallel.py
```

### Test only optimized:
```bash
python matrix_multiply_optimized.py
```

---

## 🎓 Key Learning Outcomes

1. **Parallelization:** Understanding process-level parallelism
2. **Performance Analysis:** Speedup, efficiency, and scaling
3. **Optimization:** Cache locality and memory hierarchy
4. **AI Connection:** How HPC enables modern deep learning

---

## 📝 For Your Report

Include these sections:

1. **Introduction:** Why matrix multiplication matters for AI
2. **Methods:** Description of parallel and optimization strategies
3. **Results:** Your speedup graphs and efficiency analysis
4. **Discussion:** 
   - Why efficiency < 100%?
   - How does matrix size affect performance?
   - Real-world AI implications
5. **Conclusion:** Summary of findings

The `benchmark.py` script generates most of this automatically!

---

## ⏱️ Time Required

- Setup: 5 minutes
- Running benchmarks: 2-3 minutes
- Total: **Under 10 minutes** to get complete results

---

## 🎉 Success Criteria

You've successfully completed the project if you can:
1. ✅ Run the benchmark without errors
2. ✅ Generate performance graphs
3. ✅ Show speedup > 1.5x with parallelization
4. ✅ Explain why these techniques accelerate AI computing

---

## 💡 Tips for Presentation

1. Start with the AI motivation (why matrix multiplication matters)
2. Show your performance graphs prominently
3. Discuss the tradeoffs (speedup vs efficiency)
4. Connect back to real AI applications (training large models)
5. Mention Apple M5 optimizations (Accelerate framework)

---

## 📚 Additional Resources

- NumPy Documentation: https://numpy.org/doc/
- Numba Documentation: https://numba.pydata.org/
- Matrix Multiplication in Neural Networks: [Various online resources]

---
