#pragma once
#ifndef CLAUSE_H
#define CLAUSE_H

#define NORM_EVAL_POINT_NUM 1000

#include "z3++.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <string.h>
#include <sstream>
#include <unordered_map>
#include <assert.h>
#include <numeric>

using namespace std;
using namespace z3;

typedef pair< vector<vector <int> >, vector < vector <int> >> MatInfo;
typedef pair< vector <int> , vector <int> > IndexRelation;
typedef pair< vector <int> , vector <int> > SingleApprInfo;
typedef vector <SingleApprInfo> ApprInfo;
typedef vector <ApprInfo> ApprInfoSet;

ApprInfoSet findAFV(string prefix, double approx_error_bound, double(*fp1)(double), vector <int> MaxDifference, int startIndex,int bm_id_under_test, double approx_error_bound_ratio);
int reduceDegreePrecision(int dpsMax, vector <vector <int> > FVSet, double approx_error_bound, double(*fp1)(double), vector <int> MaxDifference, int startIndex, ApprInfoSet& lowerDegreeApprSet, double approx_error_bound_ratio, int bm_id_under_test);

class ConstructClause{
public:
    ConstructClause(int _variableNumber, int _accuracy, int _gate, int _gateCate, int _gateKind, int N, vector <int> _featureVector, vector <vector <int> > _gateTruthTable, context& c, unordered_map <string, string> _gateLibrary);
    vector <vector <IndexRelation> > AddAdditionalVariable(context& c, MatInfo IndexMatrix, vector <int> largestCubeVector, vector <int> featureVector, vector <int> toBeAssignedCV1, int coarseGrain1, vector <int> toBeAssignedCV2, int coarseGrain2);

    void ConstructMainClause(ofstream& outputfile, solver& s);
//    void ConstructMainClauseWithCutSet(ofstream& outputfile, vector < vector <int> > DAG, vector <int> cutEdge);
//    void ConstructMainClauseWithFence(ofstream& outputfile, vector < vector <int> > DAG);

    void SymmetryBreakingConstraints(ofstream& outputfile);
    void MaxFanout(ofstream& outputfile, int maxFanout);
    void MaxDepthConstraint(ofstream& outputfile, int maxFenceDepth, vector < vector <int> > DAG);

    void Construct1stClause(ofstream& outputfile, context& c, solver& s);

    MatInfo PrepareForConstruct2ndClause();

    void Construct2ndClause(ofstream& outputfile, vector < vector <int> > indexOfXn_mInG, vector <int> largestCubeVector, vector <int> cubeMat);

    void ConstructFVClause(string prefix, ofstream& outputfile, vector <int> featureVector, vector <int> largestCubeVector, vector<int> toBeAssignedCV, vector <IndexRelation> AllVarRelation, int coarseGrain, MatInfo IndexMatrix);

    void ConstructFVClauseWithoutESPRESSO(string prefix, ofstream& outputfile, vector <int> featureVector, vector <int> largestCubeVector, vector<int> toBeAssignedCV1, vector <vector <IndexRelation> > AllVarRelation, int coarseGrain1, MatInfo IndexMatrix, vector <int> toBeAssignedCV2, int coarseGrain2, context& c, solver& s);

    void GateLibrary(ofstream& outputfile, context& c, solver& s);

    void GateIndexConstraint(ofstream& outputfile, int gateKind, vector <int> gateArea);

    void GateIndexConstraintZ3(ofstream& outputfile, int gateKind, context& c, solver& s);

    void areaConstraintZ3(ofstream& outputfile, int gateKind, vector <double> gateArea, double maxArea, context& c, solver& s);

    string SATSolver(int tryNum, int maxSolutionNum,string prefix, ofstream& outputfile, vector < vector <int> > indexOfXn_mInG);

    int SMTSolver(context& c, solver& s, int maxIterationPerRun, vector < vector <int> > indexOfXn_mInG, vector <double> gateArea, unsigned timeBound);

//    std::thread t1(context& c, solver& s, int maxIterationPerRun, vector < vector <int> > indexOfXn_mInG, vector <double> gateArea){
//    	return std::thread(&ConstructClause::SMTSolver, this, c, s, maxIterationPerRun, indexOfXn_mInG, gateArea);
//    }

    int PrintCircuit(string prefix, vector < vector <int> > indexOfXn_mInG);

    double PrintVerifyCircuitZ3(char* outputDir, int bm_id_under_test, model& m, vector < vector <int> > indexOfXn_mInG, vector <double> gateArea, int solutionNum, vector <int>& areaSet);

    void VerifyCircuit(string prefix, int solutionNum);
    void MappingAfterSAT(string prefix, int solutionNum);

    int variableNumber;
    int accuracy;
    int gate;
    int gateCate;
    int gateKind;
    int N;
    vector <int> featureVector;
    vector < vector <int> > X_known;
    vector < vector <int> > X_unknown;
    vector < vector <int> > AssMatrix;
    vector < vector <int> > S;
    vector < vector <int> > f;
    vector < vector <int> > GateIndex;
    vector <vector <int> > gateTruthTable;
    vector <int> YVariable;
    vector <int> ZVariable;
    vector <int> SumColumn;
    unordered_map <string, string> gateLibrary;

    vector < vector <expr> > X_Unknown_Z3;
    vector < vector <expr> > S_Z3;
    vector < vector <expr> > f_Z3;
    vector < vector <expr> > GateIndex_Z3;
    vector < vector <expr> > GateIndex_Z3_INT;
    vector <expr> Y_Variable_Z3;
    vector <expr> Z_Variable_Z3;
};

void dfs(int n, int k, int index, vector<int>& path, vector <vector <int> >& res);
vector <vector <int> > combine(int n, int k);
vector <vector <int>> ProduceBinComb(int n, int k);
int Bin_To_Dec(int b, int c, int d, int e);
void q(int n, int m, int i, int level, int k);
vector < vector <int> > GenerateFence(int gateNum);

int DetermineValidFV(int degree, int accuracy, vector <int> testVector);
vector <int> FindMaxFV(int degree, int accuracy);

double get_L2_norm_of_target_function(double(*fp1)(double), int evaluating_point_number);
double get_feature_vec_approx_error(vector<int> feature_vec, int precision, double(*fp_target_func)(double), int evaluating_point_number);
vector <double> feature_vec_2_Bern_coef_vec_converter(vector<int> feature_vec, int accuracy);
double my_Bernstein_polynomial(double x, vector<double> Bern_coef_vec);
long long int nchoosek(int n, int k);
double get_L2_norm_of_Bern_polynomial_vs_target_function(double(*fp1)(double), double(*fp2)(double, vector<double>), vector<double> Bern_coef_vec, int evaluating_point_number);

#endif
