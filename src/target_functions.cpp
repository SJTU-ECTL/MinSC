//#include "include.h"
//#include "define.h"

#define PI 3.141592654

#include <cmath>
#include "math.h"

//===========================================================
// add/modify the definition of user-defined target functions here

// for example:
double user_defined_target_function(double x) {
    return sin(x);
}


//===========================================================


double bm1_target_func(double x) {
     return sin(x);
}

double bm2_target_func(double x) {
    return cos(x);
}

double bm3_target_func(double x) {
    return tanh(x);
}

double bm4_target_func(double x) {
    return exp(-x);
}

double bm5_target_func(double x) {
    return log(x + 1);
}

double bm6_target_func(double x) {
    return pow(x, 2.2);
}

double bm7_target_func(double x) {
    return sin(PI * x) / PI;
}

double bm8_target_func(double x) {
    return exp(-2 * x);
}
double bm9_target_func(double x) {
    return 1 / (1 + x);
}

double bm10_target_func(double x) {
    return tanh(PI * x);
}

double bm11_target_func(double x) {
    return pow(x, 0.45);
}

double bm12_target_func(double x) {
    return sqrt(x);
}

double bm13_target_func(double x) {
    return tan(x/2);
}

double bm14_target_func(double x) {
    return x*log(x/2)+1;
}



