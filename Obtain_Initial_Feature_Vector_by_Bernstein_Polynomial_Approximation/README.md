# Obtain_Initial_Feature_Vector_by_Bernstein_Polynomial_Approximation (input preparation for MinSC) 

This program is supposed to be run before MinSC to prepare for its input files regarding to the target function to be approximated.

This program implements the method proposed in [1] to realize the closest approximation of a target function with a degree-n Bernstein polynomial. Moreover, given a precision parameter m, it also finds the corresponding feature vector (the same as `problem vector` in [2]) of the Bernstein polynomial.

For each target function, this program outputs its feature vectors with different degrees and precisions.

The related reference papers are:
- [1]: An Architecture for Fault-Tolerant Computation with Stochastic Logic (Qian et. al., 2011)
- [2]: Cube Assignment for Stochastic Circuit Synthesis (Peng and Qian, 2018)
- [3]: MinSC: An Exact Synthesis-Based Method for Minimal-Area Stochastic Circuits under Relaxed Error Bound (Xuan Wang, Zhufei Chu, and Weikang Qian, ICCAD 2021)

## Requirements

To run this program, Matlab2016 or later versions are recommended. The program is implemented for Matlab in Windows OS. If it is run in Linux, the directory format in the scripts shall be modified.

## Usage
### Input
Please run `main.m`, the parameters are:
- `bm_id` : the index of benchmark functions;
   ```
   target function ID    target function
    -------------------------------------
    1                     sin(x)
    2                     cos(x)
    3                     tanh(x)
    4                     exp(-x)
    5                     log(x+1)
    6                     x^2.2
    7                     sin(Pi*x)/Pi
    8                     exp(-2x)
    9                     1/(x+1)
    10                    tanh(Pi*x)
    11                    x^0.45
    12                    sqrt(x)
    13                    tan(x/2)
    14                    x*ln(x/2)+1
   ```
### Output
  The results are stored in the file `.\output_dir\Input_files_for_MinSC\bm_id.txt`. To run MinSC main program, please copy this file to the path `..\..\MinSC_main_program\FVnm`.




