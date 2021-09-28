# MinSC Overview

This project implements the exact-synthesis method to find an area-optimal SC circuit with the relaxed error bound over the target function in our paper [1].

Related papers:
- [1]: MinSC: An Exact Synthesis-Based Method for Minimal-Area Stochastic Circuits under Relaxed Error Bound (Xuan Wang, Zhufei Chu, and Weikang Qian, ICCAD 2021)

The following two directories provide the two major programs.

- [Obtain_Initial_Feature_Vector_by_Bernstein_Polynomial_Approximation](https://github.com/SJTU-ECTL/MinSC/tree/master/Obtain_Initial_Feature_Vector_by_Bernstein_Polynomial_Approximation) is supposed to be run before MinSC program to prepare the input file for it.

- [MinSC_main_program (on 64-bit Linux)](https://github.com/SJTU-ECTL/MinSC/tree/master/MinSC_main_program) is the program for the MinSC method proposed in our paper [1]. 

Please refer to `README.md` in both directories for more details.

If you have any questions or suggestions, please feel free to eamil to xuan.wang@sjtu.edu.cn, thanks!
