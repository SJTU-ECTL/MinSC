# MinSC: An Exact Synthesis-Based Method for Minimal-Area Stochastic Circuits under Relaxed Error Bound

This project implements the exact-synthesis method to find an area-optimal SC circuit with the relaxed error bound over the target function.

Related papers:
- [1]: MinSC: An Exact Synthesis-Based Method for Minimal-AreaStochastic Circuits under Relaxed Error Bound (Xuan Wang, Zhufei Chu, and Weikang Qian, ICCAD 2021)

Reference papers:
- [2]: Stochastic Circuit Synthesis by Cube Assignment (Xuesong Peng, and Weikang Qian, TCAD 2018)

## Requirements

- OS: 64-bit Linux
- gcc
- g++
- EDA logic synthesis tools: [ABC](http://people.eecs.berkeley.edu/~alanmi/abc/), [MVSIS](https://github.com/sterin/mvsis) executable files
- SMT Solver: [Z3](https://github.com/Z3Prover/z3)

## Important Notes

- Since the program requires the EDA tools [ABC](http://people.eecs.berkeley.edu/~alanmi/abc/),[MVSIS](https://ptolemy.berkeley.edu/projects/embedded/mvsis/), and SMT Solver [Z3](https://github.com/Z3Prover/z3), please download the appropriate executable files or compile the source codes in your OS. 
 - For ABC, suppose the absolute directory containing the executable file `abc` is `<abc_exe_absolute_directory>`. Then, before compiling the MinSC program, please execute command: ` cp abc_exe_absolute_directory /usr/bin/abc`, which ensures ABC can run in an arbitrary path.
 - For MVSIS, suppose the absolute directory containing the executable file `abc` is `<mvsis_exe_absolute_directory>`. Then, before compiling the MinSC program, please execute command: ` cp mvsis_exe_absolute_directory /usr/bin/mvsis`, which ensures MVSIS can run in an arbitrary path.
  
## Getting Started
### Configuration in Project
- Install Z3 at a self-defined path `Z3_Path`
- Set up a C++ project;
- Add the source files and header files in the folder `src/`;
- Configure the library path of Z3 for this project:

  Project->Property->
  - GCC C++ Complier->Includes->Include paths->Add->: `Z3_Path\src\api`;
  - GCC C++ Complier->Includes->Include paths->Add->: `Z3_Path\src\api\c++`;
  - GCC C++ Linker->Library search path(-L)->Add： `Z3_Path\build`;
  - GCC C++ Linker->Libraries(-l)->Add： `z3`;

### Input Format
- `approx_error_bound_ratio` : the error ratio which controls the error bound of approximation. In our paper, we set error ratio as 0.02 and 0.05, respectively.
- `bm_id_under_test` : the index of benchmark functions;
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
   
## Program Organization

```
<program_dir>
| readme.md
|----src
|     |----(source files)
|----gate_library
|     |----mcnc.genlib
|     |----gate.txt(the gate library after extending mcnc.genlib)
|----FVnm
|     |----1.txt
|     |----2.txt
|     |----(and so on)
|----temp_dir
|     |----(temporary files)
|----output_dir
|     |----error_ratio2
|     |     |----bm1
|     |     |----bm2
|     |     |----(and so on)
|     |----error_ratio5
|     |     |----bm1
|     |     |----bm2
|     |     |----(and so on)
|----examples_demo_results
|     |----error_ratio2
|     |     |----bm1
|     |     |----bm2
|     |     |----(and so on)
|     |----error_ratio5
|     |     |----bm1
|     |     |----bm2
|     |     |----(and so on)
```

- `src`: contains all source files and header files.
- `gate_library`: contains the mcnc.genlib and gate.txt. The file gate.txt is the gate library after extending mcnc.genlib. For a gate with the fanin number less than4, it is extended to a 4-input gate with some fake fanins.
- `FVnm`: contains the initial feature vectors of different degrees and precisions for all benchmark functions.
   Format of each row in the `bm_id_under_test`.txt file:
```
   degree n precision m feature vector
```
- `temp_dir`: contains temporary files generated during the running of the program.
- `output_dir`: contains two sub-folders, i.e., `error_ratio2` and `error_ratio5`. `error_ratio2` contains the output files for all the benchmarks with error ratio 0.02 in each sub-folder such as `bm1`, `bm2`, etc., while `error_ratio5` contains the output files for all the benchmarks with error ratio 0.05 used in each sub-folder such as `bm1`, `bm2`, etc..
  The output files are:
  - `<bm_name>-bestSol_summary.txt`: overall summary of the best solution with minimal area for `<bm_name>`.
  - `<bm_name>-solution<num>.v`: gate-level Verilog file for the solution with number `<num>`.
- `examples_demo_results`: contains output results for all benchmarks with different error ratios. It contains two sub-folders, i.e., `error_ratio2` and `error_ratio5`. `error_ratio2` contains the output files for all the benchmarks with error ratio 0.02 used in our paper [1] in each sub-folder such as `bm1`, `bm2`, etc., while `error_ratio5` contains the output files for all the benchmarks with error ratio 0.05 used in our paper [1] in each sub-folder such as `bm1`, `bm2`, etc..

## Speedup Techniques (POA & MGS)
- To speed up the solving process, for the error ratio 0.02, we apply POA and MGS techniques to the 5th, 8th, 11th, 12th, and 13th functions.  For the error ratio   0.05, these two techniques are applied to the 13th and 14th functions.
- For these target functions with error ratio 0.02, in the demo experiments, given target function ID `bm_id_under_test`, we give the detailed POA and MGS information in the path `DCV_MGS/error_ratio2/bm_id_under_test.txt`.
- For these target functions with error ratio 0.05, in the demo experiments, given target function ID `bm_id_under_test`, we give the detailed POA and MGS information in the path `DCV_MGS/error_ratio5/bm_id_under_test.txt`.
- The format of the content:
```
The first row: For POA technique, choose which level of the solution tree in Peng's method [2] for each ASCP (The index is from 1).
The second row: For POA technique, choose which cube in the current level of the solution tree in Peng's method[2] for each ASCP (The index is from 0).
The thrid row: For MGS technique, the value of the first coarse granularity of each ASCP.
The fourth row: For MGS technique, the value of the second coarse granularity of each ASCP.
Following rows: the feature vectors corresponding to each ASCP with different coarse granularities.
```
For example, for the 12th function with error ratio 0.02, it has 2 ASCPs, i.e., ASCP1 and ASCP2. The content of the file `DCV_MGS/error_ratio2/12.txt` is:
```
2 2
1 1
2 2
1 1
0 2 2 0
0 2 2 0
0 0 0 0
0 0 0 0
```
where the first line denotes that level of the solution tree we choose in Peng's method for ASCP1 and ASCP2 are both 2. The second line denotes that we choose the second cube in the current level of the solution tree in Peng's method for both ASCPs. The thrid line denotes that the value of the first coarse granularity for ASCP1 and ASCP2 are both 2. The fourth line denotes that the value of the second coarse granularity for ASCP1 and ASCP2 are both 1. The fifth line and the sixth line denote that the feature vector corresponding to ASCP1 and ASCP2 with the the first coarse granularity (i.e., 2) are both \[0,2,2,0\]. The seventh line and the eighth line denote that the feature vector corresponding to ASCP1 and ASCP2 with the the second coarse granularity (i.e., 1) are both \[0,0,0,0\].
