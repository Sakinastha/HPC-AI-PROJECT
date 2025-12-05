# HPC Project: AI-Accelerating Computing with Parallel Matrix Operations

**Course:** High Performance Computing (HPC)  
**Topic:** Accelerating AI Matrix Multiplication using HPC Techniques  


---

##  Project Overview

This project demonstrates how High Performance Computing techniques can accelerate the fundamental operations in AI and deep learning. Matrix multiplication is the core computational operation in neural networks, and optimizing it has direct impact on training and inference speed.

### Key Implementations:
1. **Serial Baseline** - Standard NumPy implementation
2. **Parallel Processing** - Multiprocessing with row-block distribution
3. **Cache Optimization** - Blocked matrix multiplication with Numba JIT

---

##  Quick Start (3-Step Setup)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Complete Benchmark
```bash
python benchmark.py
```

### Launch live dashboard
streamlit run display_app.py

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


##  Technical Details

### 1. Serial Implementation (Baseline)
- Uses NumPy's `np.dot()` which leverages Apple's Accelerate framework
- Provides baseline for speedup calculations


### 2. Parallel Implementation
**Strategy:** Row-block distribution across processes
- Divides matrix A into horizontal blocks
- Each process computes its block independently
- Uses Python's `multiprocessing` module
- Tests with 1, 2, 4, and all available cores


##  Expected Results

On my MacBook, I saw:

### Speedup Examples (1024x1024 matrix):
- **2 processes:** ~1.5-1.8x speedup
- **4 processes:** ~2.5-3.0x speedup
- **All cores:** ~3.5-5.0x speedup
- **Cache-blocked:** ~1.5-2.5x speedup (plus scales with parallelization)

### Efficiency:
- Efficiency typically 60-80% (varies with matrix size)
- Larger matrices show better efficiency
- Communication overhead affects smaller matrices



##  Troubleshooting

### If you get "numba not found":
```bash
pip install numba
```

### If plots don't show:
The script auto-saves to PNG. Check `hpc_performance_results.png`

### For macbook specific:
- No CUDA/OpenMP needed (not supported on Mac anyway)
- NumPy automatically uses Apple's Accelerate framework
- Multiprocessing works perfectly on Apple Silicon

---

##  Running Individual Tests

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





