#include "clause.hpp"

// DFS, the sub-function of Calculate Cn_k
void dfs(int n, int k, int index, vector<int>& path, vector <vector <int> >& res) {
    if (path.size() == k) {
        res.push_back(path);
        return;
    }
    for (unsigned int i = index; i <= n - (k - path.size()) + 1; ++i) {
        path.push_back(i);
        dfs(n, k, i + 1, path, res);
        path.pop_back();
    }
}

// Calculate Cn_k
vector <vector <int> > combine(int n, int k) {
    vector <vector <int> > res;
    if (k <= 0 || n < k) {
        return res;
    }
    vector<int> path;
    dfs(n, k, 1, path, res);
    return res;
}

// Convert four binary values to a decimal value
int Bin_To_Dec(int b, int c, int d, int e) {
    return b * (int)pow(2, 3) + c * (int)pow(2, 2) + d * (int)pow(2, 1) + e * (int)pow(2, 0);
}

vector <vector <int>> ProduceBinComb(int n, int k){
    vector< vector<int> > comb;
    comb = combine(n, k);
    for (auto p = 0; p < int(comb.size()); p++) {
        for (auto q = 0; q < int(comb[p].size()); q++) {
            comb[p][q] -= 1;
            //            cout << comb[p][q] << " ";
        }
        //        cout << endl;
    }

    //    cout << "The number of different combinations for combYi: " << comb.size() << endl;

    vector <int> sizeN(n);
    for (auto p = 0; p < n; p++) {
        sizeN[p] = p;
    }


    // Determine the SOP matrix for combinations
    vector <vector <int>> SOPMat(comb.size(), vector <int>(n, 0));
    for (auto p = 0; p < int(comb.size()); p++) {
        for (auto q = 0; q < int(comb[p].size()); q++) {
            auto findRes = find(sizeN.begin(), sizeN.end(), comb[p][q]);
            int index = findRes - sizeN.begin();
            SOPMat[p][index] = 1;
        }
    }

    // Output the pla format of SOP matrix
    //    cout << "The PLA format of SOP matrix:" << endl;
    //    for (auto p = 0; p < int(SOPMat.size()); p++) {
    //        for (auto q = 0; q < int(SOPMat[p].size()); q++) {
    //            cout << SOPMat[p][q];
    //        }
    //        cout << endl;
    //    }
    //    cout << endl << endl;

    return SOPMat;
}

vector < vector <int> > fenceFamily;
vector <int> path;
//Use an array to record the previous values that need to be output repeatedly during the recursion process
int set[100];

vector < vector <int> > GenerateFence(int gateNum) {

    // Used to determine whether to recurse to the deepest point in the recursion process
    int depth = gateNum, colMax = gateNum, level = gateNum;

    q(gateNum, colMax, 0, level, depth);
    return fenceFamily;
}

//This function represents the use of an integer not greater than m to split n, i is used to represent the length of
// the number of records that already exist in the set array
void q(int n, int m, int i, int level, int k)
{
    if (n == k && n != m)
    {
        //At this time, the recursive stack has returned to the top level of a branch
        //Reset the counter i to 0
        // cout << endl;
        //        for(int j = i; j < level-1; j++){
        //            path.push_back(0);
        //        }
        if (!path.empty()) {
            std::sort(path.begin(), path.end());
            //            featureVecPartition.push_back(path);
            do
            {
                if(path.back() == 1){
                    int flag = 1;
                    for(auto x = 0; x < int(path.size()) - 1; x++){
                        int sum = 0;
                        //                        if(path[x] > 3){
                        //                            flag = 0;
                        //                            break;
                        //                        }
                        for(auto y = x+1; y < int(path.size()); y++){
                            sum = sum + path[y];
                        }
                        if(path[x] > 2 * sum){
                            flag = 0;
                            break;
                        }
                    }
                    if(flag == 1){
                        fenceFamily.push_back(path);
                    }
                }
            } while (next_permutation(path.begin(), path.end()));
        }
        path.clear();
        i = 0;
    }
    if (n == 1)
    {
        path.push_back(1);
        //        for(int j = i; j < level-1; j++){
        //            path.push_back(0);
        //        }
        // printf("1 ");

        if (!path.empty()) {
            std::sort(path.begin(), path.end());
            //            featureVecPartition.push_back(path);
            do
            {
                if(path.back() == 1){
                    int flag = 1;
                    for(auto x = 0; x < int(path.size()) - 1; x++){
                        int sum = 0;
                        //                        if(path[x] > 3){
                        //                            flag = 0;
                        //                            break;
                        //                        }
                        for(auto y = x+1; y < int(path.size()); y++){
                            sum = sum + path[y];
                        }
                        if(path[x] > 2 * sum){
                            flag = 0;
                            break;
                        }
                    }
                    if(flag == 1){
                        fenceFamily.push_back(path);
                    }
                }
            } while (next_permutation(path.begin(), path.end()));
        }
        path.clear();
        return;
    }
    else if (m == 1)
    {
        // When m is 1, it means to output n 1s
        for (int i = 0; i < n - 1; i++) {
            // printf("1+");
            path.push_back(1);
        }
        // printf("1 ");
        path.push_back(1);
        //        for(int j = i; j < level-1; j++){
        //            path.push_back(0);
        //        }

        if (!path.empty()) {
            std::sort(path.begin(), path.end());
            //featureVecPartition.push_back(path);
            do
            {
                if(path.back() == 1){
                    int flag = 1;
                    for(auto x = 0; x < int(path.size()) - 1; x++){
                        int sum = 0;
                        //                        if(path[x] > 3){
                        //                            flag = 0;
                        //                            break;
                        //                        }
                        for(auto y = x+1; y < int(path.size()); y++){
                            sum = sum + path[y];
                        }
                        if(path[x] > 2 * sum){
                            flag = 0;
                            break;
                        }
                    }
                    if(flag == 1){
                        fenceFamily.push_back(path);
                    }
                }
            } while (next_permutation(path.begin(), path.end()));
        }
        path.clear();
        return;
    }
    if (n < m)
    {
        q(n, n, i, level, k);
    }
    if (n == m)
    {
        //When n is equal to m, it reaches a leaf of this recursive summation. At this time,
        // it needs to output one more space, which means the next output is another leaf
        // printf("%d ", n);
        path.push_back(n);
        //        for(int j = i; j < level-1; j++){
        //            path.push_back(0);
        //        }

        if (!path.empty()) {
            std::sort(path.begin(), path.end());
            // featureVecPartition.push_back(path);
            do
            {
                if(path.back() == 1){
                    int flag = 1;
                    for(auto x = 0; x < int(path.size()) - 1; x++){
                        int sum = 0;
                        //                        if(path[x] > 3){
                        //                            flag = 0;
                        //                            break;
                        //                        }
                        for(auto y = x+1; y < int(path.size()); y++){
                            sum = sum + path[y];
                        }
                        if(path[x] > 2 * sum){
                            flag = 0;
                            break;
                        }
                    }
                    if(flag == 1){
                        fenceFamily.push_back(path);
                    }
                }
            } while (next_permutation(path.begin(), path.end()));
        }
        path.clear();

        //Before outputting another leaf recursively, output the previously recorded numbers on the leaf together
        for(int j = 0; j < i; j++) {
            // printf("%d+", set[j]);
            path.push_back(set[j]);
        }
        q(n, m - 1, i, level, k);

    }
    if (n > m)
    {
        //If n is greater than m, use m as the decomposition
        // printf("%d+", m);
        path.push_back(m);
        //Record the value of m as the trunk node and increase i
        set[i++] = m;
        //Recursively output the decomposition after m
        q(n - m, m, i, level, k);
        //After the recursion is completed, the array record needs to be backed up by one and back to the previous node
        i--;
        //Execute another branch and output the recorded data before the next recursion
        for (int j = 0; j < i; j++) {
            // printf("%d+", set[j]);
            path.push_back(set[j]);
        }
        //Recursively output another branch situation
        q(n, m - 1, i, level, k);
    }
}

long long int nchoosek(int n, int k) {
    if ((n < k) || (k < 0)) return 0;
    long long int ret = 1;
    for (int i = 1; i <= k; ++i) {
        ret *= n--;
        ret /= i;
    }
    return ret;
}

double my_Bernstein_polynomial(double x, vector<double> Bern_coef_vec) {
    int len = int(Bern_coef_vec.size());
    int n = len - 1;
    double result = 0;
    for (int i = 0; i < len; i++) {
        double Bern_coef = Bern_coef_vec[i];
        double n_choose_i = double(nchoosek(n, i));
        double term = Bern_coef * n_choose_i * pow(x, i) * pow(1 - x, n - i);
        result += term;
    }
    return result;
}

double get_L2_norm_of_target_function(double(*fp1)(double), int evaluating_point_number) {
    int n = evaluating_point_number;
    double step = 1 / double(n);
    double sum = 0;
    for (int i = 1; i <= n; i++) {
        double v = double(i * step);
        double r1 = (*fp1)(v);
        sum += pow(r1, 2);
    }
    double result = sqrt(sum / double(n));
    return result;
}

vector<double> feature_vec_2_Bern_coef_vec_converter(vector<int> feature_vec, int accuracy) {
    //    int m = accuracy;
    int len = int(feature_vec.size());
    int n = len - 1;
    vector<double> Bern_coef_vec(len, 0);
    for (int i = 0; i < len; i++) {
        Bern_coef_vec[i] = (double(feature_vec[i]) / pow(2, accuracy)) / double(nchoosek(n, i));
    }
    return Bern_coef_vec;
}

double get_L2_norm_of_Bern_polynomial_vs_target_function(double(*fp1)(double), double(*fp2)(double, vector<double>), vector<double> Bern_coef_vec, int evaluating_point_number) {
    int n = evaluating_point_number;
    double step = 1 / double(n);
    double sum = 0;
    for (int i = 1; i <= n; i++) {
        double v = double(i * step);
        double r1 = (*fp1)(v);
        double r2 = (*fp2)(v, Bern_coef_vec);
        sum += pow((r1 - r2), 2);
    }
    double result = sqrt(sum / double(n));
    return result;
}

double get_feature_vec_approx_error(vector<int> feature_vec, int accuracy, double(*fp_target_func)(double), int evaluating_point_number) {
    vector <double> Bern_coef_vec = feature_vec_2_Bern_coef_vec_converter(feature_vec, accuracy);
    double(*fp_Bern_poly)(double, vector <double>);
    fp_Bern_poly = &my_Bernstein_polynomial;
    double norm = get_L2_norm_of_Bern_polynomial_vs_target_function(fp_target_func, fp_Bern_poly, Bern_coef_vec, evaluating_point_number);
    return norm;
}

vector <int> FindMaxFV(int degree, int accuracy){
    vector <int> maxFV;
    if(degree == 0){
        maxFV.push_back(1*pow(2,accuracy));
    }

    if(degree == 1){
        maxFV.push_back(1*pow(2,accuracy));
        maxFV.push_back(1*pow(2,accuracy));
    }

    else if(degree == 2){
        maxFV.push_back(1*pow(2,accuracy));
        maxFV.push_back(2*pow(2,accuracy));
        maxFV.push_back(1*pow(2,accuracy));
    }

    else if(degree == 3){
        maxFV.push_back(1*pow(2,accuracy));
        maxFV.push_back(3*pow(2,accuracy));
        maxFV.push_back(3*pow(2,accuracy));
        maxFV.push_back(1*pow(2,accuracy));
    }

    else if(degree == 4){
        maxFV.push_back(1*pow(2,accuracy));
        maxFV.push_back(4*pow(2,accuracy));
        maxFV.push_back(6*pow(2,accuracy));
        maxFV.push_back(4*pow(2,accuracy));
        maxFV.push_back(1*pow(2,accuracy));
    }

    else if(degree == 5){
        maxFV.push_back(1*pow(2,accuracy));
        maxFV.push_back(5*pow(2,accuracy));
        maxFV.push_back(10*pow(2,accuracy));
        maxFV.push_back(10*pow(2,accuracy));
        maxFV.push_back(5*pow(2,accuracy));
        maxFV.push_back(1*pow(2,accuracy));
    }

    else if(degree == 6){
        maxFV.push_back(1*pow(2,accuracy));
        maxFV.push_back(6*pow(2,accuracy));
        maxFV.push_back(15*pow(2,accuracy));
        maxFV.push_back(20*pow(2,accuracy));
        maxFV.push_back(15*pow(2,accuracy));
        maxFV.push_back(6*pow(2,accuracy));
        maxFV.push_back(1*pow(2,accuracy));
    }

    else if(degree == 7){
        maxFV.push_back(1*pow(2,accuracy));
        maxFV.push_back(7*pow(2,accuracy));
        maxFV.push_back(21*pow(2,accuracy));
        maxFV.push_back(35*pow(2,accuracy));
        maxFV.push_back(35*pow(2,accuracy));
        maxFV.push_back(21*pow(2,accuracy));
        maxFV.push_back(7*pow(2,accuracy));
        maxFV.push_back(1*pow(2,accuracy));
    }

    return maxFV;
}

int DetermineValidFV(int degree, int accuracy, vector <int> testVector){
    vector <int> maxFV;
    maxFV = FindMaxFV(degree, accuracy);

    int flag = 1;
    for(auto i = 0; i < testVector.size(); i++){
        if((testVector[i] < 0) || testVector[i] > maxFV[i]){
            flag = 0;
        }
    }

    return flag;
}

int reduceDegreePrecision(int dpsMax, vector <vector <int> > FVSet, double approx_error_bound, double(*fp1)(double), vector <int> MaxDifference, int startIndex, ApprInfoSet& lowerDegreeApprSet, double approx_error_bound_ratio, int bm_id_under_test){
    int flag = 1;
    int index = flag - 1;
    int check = -1;
    for(auto i = startIndex; i < dpsMax; i++){
        index = index + i;
    }

    int reduce = dpsMax;
    int apprFVNum = 0;
    for(auto n = flag; n <= dpsMax; n++){  //degree
        int m = dpsMax - n;  //accuracy
        vector <int> MaxFV = FindMaxFV(n, m); // find the maximum value of FV for (n,m)

        vector <int> initialFV;

        for(auto t = 2; t < int(FVSet[index].size()); t++){
            initialFV.push_back(FVSet[index][t]);
        }

        index++;


        // Calculate the total number of apprFVs during search
        int cycleTotalNum = 1;
        vector <int> cycleNum{};
        for(auto j = 0; j < initialFV.size(); j++){
            int max1 = min(initialFV[j] + MaxDifference[dpsMax], MaxFV[j]);
            int min1 = max(0, initialFV[j] - MaxDifference[dpsMax]);
            int temp = max1 - min1 + 1;
            cycleNum.push_back(temp);

            cycleTotalNum = cycleTotalNum * temp;
        }

        vector <vector <int>> apprFVSet{};
        vector <int> apprFV(initialFV.size(), 0);
        for(auto q = 0; q < cycleTotalNum; q++){
            auto j = q;
            for (auto k = int(initialFV.size()) - 1; k >= 0; k--) {
                int min1 = max(0, initialFV[k] - MaxDifference[dpsMax]);
                apprFV[k] = min1 + j % cycleNum[k];
                // cout << apprFV[k] << " ";
                j /= cycleNum[k];
            }
            apprFVSet.push_back(apprFV);
        }

        // Traverse on each apprFV, find all apprFVs which satisfy error requirement
        for(auto p = 0; p < cycleTotalNum; p++){

            double approx_error = get_feature_vec_approx_error(apprFVSet[p], m, fp1, NORM_EVAL_POINT_NUM);

            // if (approx_error <= approx_error_bound) {
            //     SingleApprInfo singleApprFV{};

            //     vector < int > tempnm(2, 0);
            //     tempnm[0] = n;
            //     tempnm[1] = m;
            //     singleApprFV.first = tempnm;
            //     singleApprFV.second = apprFV;
            //     ApprFVSet.push_back(singleApprFV);

            //     cout << "ApprFV " << apprFVNum << ": ";
            //     cout << "(" << n << "," << m << "): ";
            //     for (auto m = 0; m < apprFV.size(); m++) {
            //         cout << apprFV[m] << " ";
            //     }
            //     cout << endl;

            //     apprFVNum++;
            // }

            if (approx_error <= approx_error_bound) {
                vector <int> tempApprFV(int(apprFVSet[p].size()),0);
                // initialApproximationSet.push_back(apprFVSet[p]);
//                cout << "ASCP " << apprFVNum << ": ";
//                cout << "(" << n << "," << m << "), ";
                for (auto a = 0; a < apprFVSet[p].size(); a++) {
                    tempApprFV[a] = apprFVSet[p][a];
//                    cout << apprFVSet[p][a] << " ";
                }
//                cout << endl;

                apprFVNum++;

                int changeDegree = n;
                int changeAccuracy = m;

                // Determine whether m can be reduced
                vector <int> flag(n, 1);

                for(auto a = 0; a < int(apprFVSet[p].size()); a++){
                    for(auto b = 0; b < int(flag.size()); b++){
                        if(apprFVSet[p][a] % int(pow(2, (b+1))) != 0){
                            flag[b] = 0;
                        }
                    }
                }

                for(auto s = int(flag.size())-1; s >= 0; s--){
                    if(flag[s] == 1){
                        changeAccuracy = changeAccuracy - (s+1);
                        for(auto a = 0; a < int(apprFVSet[p].size()); a++){
                            tempApprFV[a] = apprFVSet[p][a] / int(pow(2, s+1));
                        }
                        break;
                    }
                }


                while(1){
                    // Determine whether n can be reduced
                    int runNum = changeDegree;
                    vector <int> temp{};
                    for(auto a = 0; a < changeDegree; a++){
                        if(a == 0){
                            temp.push_back(tempApprFV[0]);
                        }
                        else{
                            temp.push_back(tempApprFV[a] - temp[a-1]);
                        }
                    }

                    int flag = DetermineValidFV(changeDegree-1, changeAccuracy, temp);
                    // int flag1 = 1;

                    if((temp[int(temp.size())-1] == tempApprFV[int(tempApprFV.size())-1]) && (flag == 1)){
                        changeDegree--;
                        tempApprFV.pop_back();
                        tempApprFV = temp;
                    }
                    else{
                        vector <int> tempnm(2,0);
                        tempnm[0] = changeDegree;
                        tempnm[1] = changeAccuracy;

                        int index = changeAccuracy + changeDegree - 1;

                        if(index + 1 < dpsMax){
                            reduce = index + 1;
                        }

                        SingleApprInfo appr;

                        appr.first = tempnm;
                        appr.second = tempApprFV;
                        lowerDegreeApprSet[index].push_back(appr);

                        for(auto i = 0; i < lowerDegreeApprSet.size(); i++){
                            if(!lowerDegreeApprSet[i].empty()){
                                if((i == 6) && (lowerDegreeApprSet[i].size() > 3) && (bm_id_under_test == 11) && (approx_error_bound_ratio == 0.02)){
                                    check = 1;
                                    break;
                            }
                        }
            }

                        // cout << changeDegree << "," << changeAccuracy << " ";
                            // for(auto t = 0; t < int(tempApprFV.size()); t++){
                            //     cout << tempApprFV[t] << " ";
                            // }
                            // cout << endl;

                        break;
                    }
                }
            }

            if(reduce < dpsMax || check > 0){
                break;
            }
        }
        if(reduce < dpsMax || check > 0){
            break;
        }
    }
    return reduce;
}

ApprInfoSet findAFV(string prefix, double approx_error_bound, double(*fp1)(double), vector <int> MaxDifference, int startIndex,int bm_id_under_test, double approx_error_bound_ratio){
	char cmd[100];
	strcpy(cmd, "../FVnm/");
	strcat(cmd, std::to_string(bm_id_under_test).data());
	strcat(cmd, ".txt");
	ifstream in;
	in.open(cmd);
//    in.open("./FVnm/cos(x).txt");
    auto line = string();
    int count = 0;

    ApprInfoSet lowerDegreeApprSet(10);

    // Read FVs of different n and m
    vector <vector <int> > FVSet{};
    vector <vector <int> > initialApproximationSet{};

    if(in){
        while(getline(in,line)){
            stringstream ss(line);
            vector <int> res;
            res.clear();
            int i;

            while(ss >> i){
                res.push_back(i);
            }
            FVSet.push_back(res);
        }
    }

    int dpsMax = -1;
    int num = 0;

    int dpsUB = 15;
    for(auto i = startIndex; i < dpsUB; i++){  // the sum of n and m
        // Given a specific sum of n and m, tranverse on all (n,m) pairs
        for(auto n = 1; n <= i; n++){  //degree
            int m = i - n;  //accuracy

            vector <int> initialFV;

            for(auto t = 2; t < int(FVSet[num].size()); t++){
                initialFV.push_back(FVSet[num][t]);
            }

            double approx_error = get_feature_vec_approx_error(initialFV, m, fp1, NORM_EVAL_POINT_NUM);

            if (approx_error <= approx_error_bound) {
                dpsMax = n+m;
                break;
            }
            num++;

        }

        if(dpsMax > 0){
            break;
        }
    }

    ApprInfo ApprFVSet{};

    int reduce = reduceDegreePrecision(dpsMax,FVSet,approx_error_bound, fp1, MaxDifference, startIndex, lowerDegreeApprSet, approx_error_bound_ratio, bm_id_under_test);

    while (reduce < dpsMax){
        dpsMax = reduce;
        for(auto i = 0; i < lowerDegreeApprSet.size(); i++){
            if(!lowerDegreeApprSet[i].empty()){
                lowerDegreeApprSet[i].clear();
            }
        }
        reduce = reduceDegreePrecision(dpsMax,FVSet,approx_error_bound, fp1, MaxDifference, startIndex, lowerDegreeApprSet, approx_error_bound_ratio, bm_id_under_test);

    }
    return lowerDegreeApprSet;

}

ConstructClause::ConstructClause(int _variableNumber, int _accuracy, int _gate, int _gateCate, int _gateKind, int _N, vector <int> _featureVector, vector <vector <int> > _gateTruthTable, context& c, unordered_map <string, string> _gateLibrary) {
    variableNumber = _variableNumber;
    accuracy = _accuracy;
    gate = _gate;
    gateCate = _gateCate;
    gateKind = _gateKind;
    featureVector = _featureVector;
    N = _N;
    gateTruthTable = _gateTruthTable;
    gateLibrary = _gateLibrary;

    // Determine the values in the truth table for the input variables and input constant coefficients
    X_known = vector < vector <int> > (_variableNumber + _accuracy, vector <int>((int)pow(2, _variableNumber + _accuracy)));

    // Transform the t-th bit in the truth table into binary values, and assign them to the input variables
    //and input constant coefficients
    for (auto t = 0; t < pow(2, variableNumber + accuracy); t++) {
        auto j = t;
        for (auto i = variableNumber + accuracy - 1; i >= 0; i--) {
            X_known[i][t] = j % 2;
            //cout << X_known[i][t] << " ";
            j /= 2;
        }
        //cout << endl;
    }

//    if(variableNumber + accuracy < N){
//    	vector <int> constOne((int)pow(2, _variableNumber + _accuracy), 1);
//    	vector <int> constZero((int)pow(2, _variableNumber + _accuracy), 0);
//    	X_known.push_back(constOne);
//    	X_known.push_back(constZero);
//    }

    // Define the values in the truth table for each gate，and number them sequentially.
    X_unknown = vector < vector <int> >(_gate, vector <int>((int)pow(2, _variableNumber + _accuracy)));
    int Number = 1;
//    cout << "The X_unknown matrix:" << endl;
    for (unsigned int i = 0; i < X_unknown.size(); i++) {
        for (unsigned int j = 0; j < X_unknown[0].size(); j++) {
            X_unknown[i][j] = Number;
//            cout << X_unknown[i][j] << " ";
            Number++;
        }
//        cout << endl;
    }

    for (unsigned int i = 0; i < gate; i++) {
    	vector <expr> temp;
    	temp.clear();
		for (unsigned int j = 0; j < (int)pow(2, variableNumber + accuracy); j++) {
			temp.push_back(c.bool_const(("XUnknown" + std::to_string(i) + "," + std::to_string(j)).c_str()));
		}
		X_Unknown_Z3.push_back(temp);
    }

    // Obtain the current largest index
//    cout << "The number of X_unknown:" << Number - 1 << endl;
//    cout << endl << endl;

    AssMatrix = vector < vector <int> >(int(pow(2, _accuracy)), vector <int>((int)pow(2, _variableNumber), -1));

    // Define the selection variables s_ij
    S = vector < vector <int> >(gate, vector<int>(0));
    for (auto i = 0; i < gate; i++) {
        S[i].resize(variableNumber + accuracy + i);
    }

    // Number s_ij sequentially
//    cout << "The S matrix:" << endl;
    for (unsigned int i = 0; i < S.size(); i++) {
        for (unsigned int j = 0; j < S[i].size(); j++) {
            S[i][j] = Number;
//            cout << S[i][j] << " ";
            Number++;
        }
//        cout << endl;
    }

    for (unsigned int i = 0; i < S.size(); i++) {
    	vector <expr> temp;
    	temp.clear();
		for (unsigned int j = 0; j < S[i].size(); j++) {
			temp.push_back(c.bool_const(("S" + std::to_string(i) + "," +std::to_string(j)).c_str()));
		}
		S_Z3.push_back(temp);
     }

    // Obtain the current largest index
//    cout << "The sum number of X_unknown and S:" << Number - 1 << endl;
//    cout << endl << endl;

    // Define the local truth table for each gate operator and number them sequentially
    f = vector < vector <int> >(gate, vector <int>((int)pow(2, N)));
//    cout << "The f matrix:" << endl;
    for (unsigned int i = 0; i < f.size(); i++) {
        for (unsigned int j = 0; j < f[i].size(); j++) {
            f[i][j] = Number;
//            cout << f[i][j] << " ";
            Number++;
        }
//        cout << endl;
    }

    for (unsigned int i = 0; i < f.size(); i++) {
    	vector <expr> temp;
    	temp.clear();
		for (unsigned int j = 0; j < f[i].size(); j++) {
			temp.push_back(c.bool_const(("f" + std::to_string(i) + "," + std::to_string(j)).c_str()));
		}
		f_Z3.push_back(temp);
	}

//    cout << "The sum number of X_unknown and S and f:" << Number - 1 << endl;
//    cout << endl << endl;

    // Define the gate index for each gate and number them sequentially
    GateIndex = vector < vector <int> >(gate, vector <int>(gateCate));
//    cout << "The GateIndex matrix:" << endl;
    for (unsigned int i = 0; i < GateIndex.size(); i++) {
        for (unsigned int j = 0; j < GateIndex[i].size(); j++) {
            GateIndex[i][j] = Number;
//            cout << GateIndex[i][j] << " ";
            Number++;
        }
//        cout << endl;
    }

    for (unsigned int i = 0; i < gate; i++) {
    	vector <expr> temp;
    	temp.clear();
		for (unsigned int j = 0; j < gateKind; j++) {
			temp.push_back(c.bool_const(("GateIndex" + std::to_string(i) + "," + std::to_string(j)).c_str()));
		}
		GateIndex_Z3.push_back(temp);
	}

    for (unsigned int i = 0; i < gate; i++) {
		vector <expr> temp;
		temp.clear();
		for (unsigned int j = 0; j < gateKind; j++) {
			temp.push_back(c.int_const(("GateIndexInt" + std::to_string(i) + "," + std::to_string(j)).c_str()));
		}
		GateIndex_Z3_INT.push_back(temp);
	}

//    cout << "The sum number of X_unknown and S and f and GateIndex:" << Number - 1 << endl;
//    cout << endl << endl;


    YVariable = vector <int> {};
    ZVariable = vector <int> {};
    SumColumn = vector <int>((int)pow(2, variableNumber), 0);

//    Y_Variable_Z3.push_back();
////    Z_Variable_Z3;

//    //Establish gate library
//    gateLibrary.insert(pair <string, string> ("0001", "and2"));
//    gateLibrary.insert(pair <string, string> ("0111", "or2"));
//    gateLibrary.insert(pair <string, string> ("1110", "nand2"));
//    gateLibrary.insert(pair <string, string> ("1000", "nor2"));
//    gateLibrary.insert(pair <string, string> ("0110", "xor2a"));
//    gateLibrary.insert(pair <string, string> ("1001", "xnor2a"));
//    gateLibrary.insert(pair <string, string> ("1100", "inv1"));
//    gateLibrary.insert(pair <string, string> ("1010", "inv2"));
//    gateLibrary.insert(pair <string, string> ("0011", "buf1"));
//    gateLibrary.insert(pair <string, string> ("0101", "buf2"));
//    gateLibrary.insert(pair <string, string> ("0010", "ab_"));
//    gateLibrary.insert(pair <string, string> ("0100", "a_b"));
//    gateLibrary.insert(pair <string, string> ("1011", "a+b_"));
//    gateLibrary.insert(pair <string, string> ("1101", "a_+b"));
//    gateLibrary.insert(pair <string, string> ("0000", "constant0"));
//    gateLibrary.insert(pair <string, string> ("1111", "constant1"));
}

vector <vector <IndexRelation> > ConstructClause::AddAdditionalVariable(context& c, MatInfo IndexMatrix, vector <int> largestCubeVector, vector <int> featureVector, vector <int> toBeAssignedCV1, int coarseGrain1, vector <int> toBeAssignedCV2, int coarseGrain2){

    vector <vector <IndexRelation> > AllVarRelation;
//    cout << "Add Y Variables:" << endl;
    vector <IndexRelation> AllVarRelationYX;

    // Determine the index of first Y_variable.
    int p = int(GateIndex.size());
    int q = int(GateIndex[0].size());
    int YVarNum = GateIndex[p-1][q-1] + 1;
    int AddYVarNum = 0;
    int AddZVarNum = 0;

    // Traverse Assignment matrix to find the relationship between X_variables and Y_variables.
    for(auto i = 0; i < int(AssMatrix[0].size()); i++){
        for(auto j = 0; j < int(AssMatrix.size()); j=j+coarseGrain2){
            //            if(AssMat[j][i] == -1){
            vector <int> YvarIndex;

            // Find the location i in G[i] of columns
            int columnInGLoc = -1;
            for(auto p = 0; p < int(IndexMatrix.first.size()); p++){
                for(auto q = 0; q < int(IndexMatrix.first[p].size()); q++){
                    if(IndexMatrix.first[p][q] == i){
                        columnInGLoc = p;
                        break;
                    }
                }
                if(columnInGLoc == p){
                    break;
                }
            }

            // Store the index of Y_variable and its location in G
            YvarIndex.push_back(YVarNum);
            YvarIndex.push_back(columnInGLoc);

            YVariable.push_back(YVarNum);
//            cout << YVarNum << " ";

            Y_Variable_Z3.push_back(c.bool_const(("YVariable" + std::to_string(AddYVarNum)).c_str()));

            // Store corresponding indexes of X_variables
            vector <int> XVarIndex;
            for(auto k = i*int(pow(2,accuracy))+j; k < i * int(pow(2,accuracy))+j+coarseGrain2; k++){
                int Xindex = X_unknown[gate-1][k];
                //                int Xindex = k;
                XVarIndex.push_back(Xindex);
            }

            IndexRelation RelationYX(YvarIndex, XVarIndex);
            AllVarRelationYX.push_back(RelationYX);
            YVarNum++;
            AddYVarNum++;

            XVarIndex.clear();
            YvarIndex.clear();
            //            }
        }
    }

//    cout << endl << endl;
//    cout << "Add Z Variables:" << endl;
    vector <IndexRelation> AllVarRelationZY;
    vector <IndexRelation> AllVarRelationZX;

    int ZVarNum = GateIndex[p-1][q-1] + int(YVariable.size()) + 1;
    // Traverse Assignment matrix to find the relationship between Z_variables and Y_variables.
    for(auto i = 0; i < int(AssMatrix[0].size()); i++){
        for(auto j = 0; j < int(AssMatrix.size()); j=j+coarseGrain1){

            vector <int> ZvarIndex;

            // Find the location i in G[i] of columns for z variables
            int columnInGLoc = -1;
            for(auto p = 0; p < int(IndexMatrix.first.size()); p++){
                for(auto q = 0; q < int(IndexMatrix.first[p].size()); q++){
                    if(IndexMatrix.first[p][q] == i){
                        columnInGLoc = p;
                        break;
                    }
                }
                if(columnInGLoc == p){
                    break;
                }
            }

            // Store the index of Y_variable and its location in G
            ZvarIndex.push_back(ZVarNum);
            ZvarIndex.push_back(columnInGLoc);

            ZVariable.push_back(ZVarNum);
//            cout << ZVarNum << " ";

            Z_Variable_Z3.push_back(c.bool_const(("ZVariable" + std::to_string(AddZVarNum)).c_str()));

            // Store corresponding indexes of Y_variables
            vector <int> YVarIndex;
            for(auto k = i*int(pow(2,accuracy))+j; k < i * int(pow(2,accuracy))+j+coarseGrain1; k++){
                if(k % coarseGrain2 == 0){
                    int Yindex = YVariable[k/coarseGrain2];
                    YVarIndex.push_back(Yindex);
                }
            }


            // Store corresponding indexes of X_variables
            vector <int> XVarIndex;
            for(auto k = i*int(pow(2,accuracy))+j; k < i * int(pow(2,accuracy))+j+coarseGrain1; k++){
                int Xindex = X_unknown[gate-1][k];
                XVarIndex.push_back(Xindex);
            }

            IndexRelation RelationZY(ZvarIndex, YVarIndex);
            AllVarRelationZY.push_back(RelationZY);

            IndexRelation RelationZX(ZvarIndex, XVarIndex);
            AllVarRelationZX.push_back(RelationZX);

            ZVarNum++;
            AddZVarNum++;

            ZvarIndex.clear();
            YVarIndex.clear();
            XVarIndex.clear();
            //                        }
        }
    }

    AllVarRelation.push_back(AllVarRelationZY);
    AllVarRelation.push_back(AllVarRelationYX);
    AllVarRelation.push_back(AllVarRelationZX);


//    cout << endl << endl;

    return AllVarRelation;
}

// Construct main clause
void ConstructClause::ConstructMainClause(ofstream& outputfile, solver& s) {
    int mainClauseNum = 0;
    for (auto i = variableNumber + accuracy + 1; i <= variableNumber + accuracy + gate; i++) {
		int n0 = i - 1, j0 = N;
		// vector<vector<int> > InputComb;
		vector<vector<int> > GateInputComb;
		// Calculate C(i-1)_N, Mat stores different combinations in C(i-1)_N, where each combination is represented as a row

		//cout << "The different input combinations for gate" << i << ":" << endl;
		GateInputComb = combine(n0, j0);
//		cout << "The number of different input combinations for gate" << i << ": " << GateInputComb.size() << endl;

		// // If gate i (at level t) has two fains j and k (j < k), gate k should be on level t−1. Otherwise, delete this combination
		int q = i - (variableNumber + accuracy + 1);  // The gate index in S_ij matrix and X_unknown matrix
		// for(auto x = 0; x < int(InputComb.size()); x++){
		//     if((InputComb[x][N-1] >= minIndex[q] + 1) && (InputComb[x][N-1] <= maxIndex[q] + 1)){
		//         GateInputComb.push_back(InputComb[x]);
		//     }
		// }
		// cout << "The number of different input combinations for gate" << i << ": " << GateInputComb.size() << endl;


		for (unsigned int p = 0; p < GateInputComb.size(); p++) { // Traverse each input combination
			int j = GateInputComb[p][0], k = GateInputComb[p][1], m = GateInputComb[p][2], n = GateInputComb[p][3]; // Obtain each element in the input combination
			if ((j <= variableNumber + accuracy) && (k <= variableNumber + accuracy) && (m <= variableNumber + accuracy) && (n <= variableNumber + accuracy)) {  // If the elements both are input variables or constant coefficients
				for (int t = 0; t < (int)pow(2, variableNumber + accuracy); t++) {  // Traverse each bit in the truth table
					{
						int b = X_known[j - 1][t], c = X_known[k - 1][t], d = X_known[m - 1][t], e = X_known[n - 1][t]; // Obtain the values of t-th bit in the truth table for two elements
						int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
						for (int a = 0; a <= 1; a++) {  // Traverse on a
							if (a == 0) {
								outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -f[q][h] << " " << 0 << endl;
								mainClauseNum++;
								s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !f_Z3[q][h]);
							}
							if (a == 1) {
								outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << f[q][h] << " " << 0 << endl;
								mainClauseNum++;
								s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || f_Z3[q][h]);
							}
						}
					}
				}
			}

			if ((j <= variableNumber + accuracy) && (k <= variableNumber + accuracy) && (m <= variableNumber + accuracy) && (n > variableNumber + accuracy)) { // If 1st element both is input variable or constant coefficient, the 2nd element is a gate.
				for (int t = 0; t < pow(2, variableNumber + accuracy); t++) { // Traverse each bit in the truth table
					{
						int b = X_known[j - 1][t], c = X_known[k - 1][t], d = X_known[m - 1][t]; // Obtain the values of t-th bit in the truth table for 1st element
						int g = n - (variableNumber + accuracy + 1); // Calculate 2nd gate index in S_ij matrix
						for (auto e = 0; e <= 1; e++) {  // Traverse on e
							for (auto a = 0; a <= 1; a++) {  // Traverse on a
								if ((e == 0) && (a == 0)) {
									int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
									outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " "<< X_unknown[q][t] << " " << X_unknown[g][t] << " " << -f[q][h] << " " << 0 << endl;
									mainClauseNum++;
									s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[g][t] || !f_Z3[q][h]);
								}
								if ((e == 0) && (a == 1)) {
									int h = Bin_To_Dec(b, c, d, e);
									outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[g][t] << " " << f[q][h] << " " << 0 << endl;
									mainClauseNum++;
									s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[g][t] || f_Z3[q][h]);
								}
								if ((e == 1) && (a == 0)) {
									int h = Bin_To_Dec(b, c, d, e);
									outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[g][t] << " " << -f[q][h] << " " << 0 << endl;
									mainClauseNum++;
									s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[g][t] || !f_Z3[q][h]);
								}
								if ((e == 1) && (a == 1)) {
									int h = Bin_To_Dec(b, c, d, e);
									outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[g][t] << " " << f[q][h] << " " << 0 << endl;
									mainClauseNum++;
									s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[g][t] || f_Z3[q][h]);
								}
							}
						}
					}
				}
			}

			if ((j <= variableNumber + accuracy) && (k <= variableNumber + accuracy) && (m > variableNumber + accuracy) && (n > variableNumber + accuracy)) {
				for (int t = 0; t < pow(2, variableNumber + accuracy); t++) {
					int b = X_known[j - 1][t], c = X_known[k - 1][t];
					int x = m - (variableNumber + accuracy + 1);
					int y = n - (variableNumber + accuracy + 1);
					for (auto d = 0; d <= 1; d++) {
						for (auto e = 0; e <= 1; e++) {
							for (auto a = 0; a <= 1; a++) {
								if ((d == 0) && (e == 0) && (a == 0)) {
									int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
									outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
									" " << -f[q][h] << " " << 0 << endl;
									mainClauseNum++;
									s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || !f_Z3[q][h]);
								}

								if ((d == 0) && (e == 0) && (a == 1)) {
									int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
									outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
									" " << f[q][h] << " " << 0 << endl;
									mainClauseNum++;
									s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || f_Z3[q][h]);
								}

								if ((d == 0) && (e == 1) && (a == 0)) {
									int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
									outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
									" " << -f[q][h] << " " << 0 << endl;
									mainClauseNum++;
									s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || !f_Z3[q][h]);
								}

								if ((d == 0) && (e == 1) && (a == 1)) {
									int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
									outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
									" " << f[q][h] << " " << 0 << endl;
									mainClauseNum++;
									s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || f_Z3[q][h]);
								}

								if ((d == 1) && (e == 0) && (a == 0)) {
									int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
									outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
									" " << -f[q][h] << " " << 0 << endl;
									mainClauseNum++;
									s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || !f_Z3[q][h]);
								}

								if ((d == 1) && (e == 0) && (a == 1)) {
									int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
									outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
									" " << f[q][h] << " " << 0 << endl;
									mainClauseNum++;
									s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || f_Z3[q][h]);
								}

								if ((d == 1) && (e == 1) && (a == 0)) {
									int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
									outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
									" " << -f[q][h] << " " << 0 << endl;
									mainClauseNum++;
									s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || !f_Z3[q][h]);
								}

								if ((d == 1) && (e == 1) && (a == 1)) {
									int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
									outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
									" " << f[q][h] << " " << 0 << endl;
									mainClauseNum++;
									s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || f_Z3[q][h]);
								}

							}
						}
					}
				}
			}

			if ((j <= variableNumber + accuracy) && (k > variableNumber + accuracy) && (m > variableNumber + accuracy) && (n > variableNumber + accuracy)) {
				for (int t = 0; t < pow(2, variableNumber + accuracy); t++) {
					int b = X_known[j - 1][t];
					int v = k - (variableNumber + accuracy + 1);
					int x = m - (variableNumber + accuracy + 1);
					int y = n - (variableNumber + accuracy + 1);
					for(auto c = 0; c <= 1; c++){
						for (auto d = 0; d <= 1; d++) {
							for (auto e = 0; e <= 1; e++) {
								for (auto a = 0; a <= 1; a++) {
									if ((c == 0) && (d == 0) && (e == 0) && (a == 0)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << X_unknown[v][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
										" " << -f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || !f_Z3[q][h]);
									}

									if ((c == 0) && (d == 0) && (e == 0) && (a == 1)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[v][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
										" " << f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || f_Z3[q][h]);
									}

									if ((c == 0) && (d == 0) && (e == 1) && (a == 0)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << X_unknown[v][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
										" " << -f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || !f_Z3[q][h]);
									}

									if ((c == 0) && (d == 0) && (e == 1) && (a == 1)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[v][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
										" " << f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || f_Z3[q][h]);
									}

									if ((c == 0) && (d == 1) && (e == 0) && (a == 0)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << X_unknown[v][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
										" " << -f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || !f_Z3[q][h]);
									}

									if ((c == 0) && (d == 1) && (e == 0) && (a == 1)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[v][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
										" " << f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || f_Z3[q][h]);
									}

									if ((c == 0) && (d == 1) && (e == 1) && (a == 0)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << X_unknown[v][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
										" " << -f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || !f_Z3[q][h]);
									}

									if ((c == 0) && (d == 1) && (e == 1) && (a == 1)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[v][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
										" " << f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || f_Z3[q][h]);
									}

									if ((c == 1) && (d == 0) && (e == 0) && (a == 0)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[v][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
										" " << -f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || !f_Z3[q][h]);
									}

									if ((c == 1) && (d == 0) && (e == 0) && (a == 1)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[v][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
										" " << f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || f_Z3[q][h]);
									}

									if ((c == 1) && (d == 0) && (e == 1) && (a == 0)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[v][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
										" " << -f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || !f_Z3[q][h]);
									}

									if ((c == 1) && (d == 0) && (e == 1) && (a == 1)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[v][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
										" " << f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || f_Z3[q][h]);
									}

									if ((c == 1) && (d == 1) && (e == 0) && (a == 0)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[v][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
										" " << -f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || !f_Z3[q][h]);
									}

									if ((c == 1) && (d == 1) && (e == 0) && (a == 1)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[v][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
										" " << f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || f_Z3[q][h]);
									}

									if ((c == 1) && (d == 1) && (e == 1) && (a == 0)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[v][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
										" " << -f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || !f_Z3[q][h]);
									}

									if ((c == 1) && (d == 1) && (e == 1) && (a == 1)) {
										int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
										outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[v][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
										" " << f[q][h] << " " << 0 << endl;
										mainClauseNum++;
										s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || f_Z3[q][h]);
									}

								}
							}
						}
					}
				}
			}

			if ((j > variableNumber + accuracy) && (k > variableNumber + accuracy) && (m > variableNumber + accuracy) && (n > variableNumber + accuracy)) {
				for (int t = 0; t < pow(2, variableNumber + accuracy); t++) {
					int u = j - (variableNumber + accuracy + 1);
					int v = k - (variableNumber + accuracy + 1);
					int x = m - (variableNumber + accuracy + 1);
					int y = n - (variableNumber + accuracy + 1);
					for(auto b = 0; b <= 1; b++){
						for(auto c = 0; c <= 1; c++){
							for (auto d = 0; d <= 1; d++) {
								for (auto e = 0; e <= 1; e++) {
									for (auto a = 0; a <= 1; a++) {
										if ((b == 0) && (c == 0) && (d == 0) && (e == 0) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << X_unknown[u][t] << " " << X_unknown[v][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 0) && (c == 0) && (d == 0) && (e == 0) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[u][t] << " " << X_unknown[v][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 0) && (c == 0) && (d == 0) && (e == 1) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << X_unknown[u][t] << " " << X_unknown[v][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 0) && (c == 0) && (d == 0) && (e == 1) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[u][t] << " " << X_unknown[v][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 0) && (c == 0) && (d == 1) && (e == 0) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << X_unknown[u][t] << " " << X_unknown[v][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 0) && (c == 0) && (d == 1) && (e == 0) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[u][t] << " " << X_unknown[v][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 0) && (c == 0) && (d == 1) && (e == 1) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << X_unknown[u][t] << " " << X_unknown[v][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 0) && (c == 0) && (d == 1) && (e == 1) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[u][t] << " " << X_unknown[v][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 0) && (c == 1) && (d == 0) && (e == 0) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << X_unknown[u][t] << " " << -X_unknown[v][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 0) && (c == 1) && (d == 0) && (e == 0) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[u][t] << " " << -X_unknown[v][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 0) && (c == 1) && (d == 0) && (e == 1) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << X_unknown[u][t] << " " << -X_unknown[v][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 0) && (c == 1) && (d == 0) && (e == 1) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[u][t] << " " << -X_unknown[v][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 0) && (c == 1) && (d == 1) && (e == 0) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << X_unknown[u][t] << " " << -X_unknown[v][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 0) && (c == 1) && (d == 1) && (e == 0) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[u][t] << " " << -X_unknown[v][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 0) && (c == 1) && (d == 1) && (e == 1) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << X_unknown[u][t] << " " << -X_unknown[v][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 0) && (c == 1) && (d == 1) && (e == 1) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << X_unknown[u][t] << " " << -X_unknown[v][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 1) && (c == 0) && (d == 0) && (e == 0) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[u][t] << " " << X_unknown[v][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 1) && (c == 0) && (d == 0) && (e == 0) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[u][t] << " " << X_unknown[v][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 1) && (c == 0) && (d == 0) && (e == 1) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[u][t] << " " << X_unknown[v][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 1) && (c == 0) && (d == 0) && (e == 1) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[u][t] << " " << X_unknown[v][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 1) && (c == 0) && (d == 1) && (e == 0) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[u][t] << " " << X_unknown[v][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 1) && (c == 0) && (d == 1) && (e == 0) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[u][t] << " " << X_unknown[v][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 1) && (c == 0) && (d == 1) && (e == 1) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[u][t] << " " << X_unknown[v][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 1) && (c == 0) && (d == 1) && (e == 1) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[u][t] << " " << X_unknown[v][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 1) && (c == 1) && (d == 0) && (e == 0) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[u][t] << " " << -X_unknown[v][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 1) && (c == 1) && (d == 0) && (e == 0) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[u][t] << " " << -X_unknown[v][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 1) && (c == 1) && (d == 0) && (e == 1) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[u][t] << " " << -X_unknown[v][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 1) && (c == 1) && (d == 0) && (e == 1) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[u][t] << " " << -X_unknown[v][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 1) && (c == 1) && (d == 1) && (e == 0) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[u][t] << " " << -X_unknown[v][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 1) && (c == 1) && (d == 1) && (e == 0) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[u][t] << " " << -X_unknown[v][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

										if ((b == 1) && (c == 1) && (d == 1) && (e == 1) && (a == 0)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << X_unknown[q][t] << " " << -X_unknown[u][t] << " " << -X_unknown[v][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << -f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || !f_Z3[q][h]);
										}

										if ((b == 1) && (c == 1) && (d == 1) && (e == 1) && (a == 1)) {
											int h = Bin_To_Dec(b, c, d, e); // Convert binary values to a decimal value
											outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -S[q][m - 1] << " " << -S[q][n - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[u][t] << " " << -X_unknown[v][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
											" " << f[q][h] << " " << 0 << endl;
											mainClauseNum++;
											s.add(!S_Z3[q][j-1] || !S_Z3[q][k-1] || !S_Z3[q][m-1] || !S_Z3[q][n-1] || !X_Unknown_Z3[q][t] || !X_Unknown_Z3[u][t] || !X_Unknown_Z3[v][t] || !X_Unknown_Z3[x][t] || !X_Unknown_Z3[y][t] || f_Z3[q][h]);
										}

									}
								}
							}
						}
					}
				}
			}
		}

	}

//    cout << "The number of main clauses:" << mainClauseNum << endl;
}

//// Construct main clause
//void ConstructClause::ConstructMainClauseWithCutSet(ofstream& outputfile, vector < vector <int> > DAG, vector <int> cutEdge) {
//    int mainClauseNum = 0;
//
//    vector < vector <int> > cutSetAllCons;
//    for(auto x = 1; x < int(DAG.size()); x++){
//        vector <int> cutSetCons;
//        for(auto y = 0; y < int(DAG[x].size()); y++){
//            int p = DAG[x][y];
//            for(auto z = 0; z < int(DAG[x-1].size()); z++){
//                int q = DAG[x-1][z] + variableNumber + accuracy;
//                cutSetCons.push_back(S[p][q]);
//            }
//        }
//        cutSetAllCons.push_back(cutSetCons);
//        cutSetCons.empty();
//    }
//
//    for(auto x = 0; x < int(cutEdge.size()); x++){
//        vector<vector<int> > atLeastComb;
//        int m = int(cutSetAllCons[x].size());
//        int n = int(cutSetAllCons[x].size())-(cutEdge[x]-1);
//        atLeastComb = combine(m, n);
//
//        cout << "The number of different combinations for atLeast" << cutEdge[x] << "Comb: " << atLeastComb.size() << endl;
//
//        for (unsigned int p = 0; p < atLeastComb.size(); p++) {
//            for (unsigned int q = 0; q < atLeastComb[p].size(); q++) {
//                int i = atLeastComb[p][q]-1;
//                outputfile << cutSetAllCons[x][i] << " ";
//
//            }
//            mainClauseNum++;
//            outputfile << 0 << endl;
//        }
//
//
//        vector<vector<int> > atMostComb;
//        int l = cutEdge[x]+1;
//        atMostComb = combine(m, l);
//
//        cout << "The number of different combinations for AtMost" << cutEdge[x] << "Comb: " << atMostComb.size() << endl;
//
//
//        for (unsigned int p = 0; p < atMostComb.size(); p++) {
//            for (unsigned int q = 0; q < atMostComb[p].size(); q++) {
//                int j = atMostComb[p][q]-1;
//                outputfile << -cutSetAllCons[x][j] << " ";
//
//            }
//            outputfile << 0 << endl;
//            mainClauseNum++;
//        }
//    }
//
//    for (auto i = variableNumber + accuracy + 1; i <= variableNumber + accuracy + gate; i++) {
//        int n0 = i - 1, j0 = N;
//        //        vector<vector<int> > InputComb;
//        vector<vector<int> > GateInputComb;
//        // Calculate C(i-1)_N, Mat stores different combinations in C(i-1)_N, where each combination is represented as a row
//
//        //cout << "The different input combinations for gate" << i << ":" << endl;
//        GateInputComb = combine(n0, j0);
//        cout << "The number of different input combinations for gate" << i << ": " << GateInputComb.size() << endl;
//
//        // If gate i (at level t) has two fains j and k (j < k), gate k should be on level t−1. Otherwise, delete this combination
//        int q = i - (variableNumber + accuracy + 1);  // The gate index in S_ij matrix and X_unknown matrix
//        //        for(auto x = 0; x < int(InputComb.size()); x++){
//        //            if((InputComb[x][1] >= minIndex[q] + 1) && (InputComb[x][1] <= maxIndex[q] + 1)){
//        //                GateInputComb.push_back(InputComb[x]);
//        //            }
//        //        }
//        //        cout << "The number of different input combinations for gate" << i << ": " << GateInputComb.size() << endl;
//
//
//        for (unsigned int p = 0; p < GateInputComb.size(); p++) { // Traverse each input combination
//            int j = GateInputComb[p][0], k = GateInputComb[p][1]; // Obtain each element in the input combination
//            if ((j <= variableNumber + accuracy) && (k <= variableNumber + accuracy)) {  // If the elements both are input variables or constant coefficients
//                for (int t = 0; t < (int)pow(2, variableNumber + accuracy); t++) {  // Traverse each bit in the truth table
//                    {
//                        int b = X_known[j - 1][t], c = X_known[k - 1][t]; // Obtain the values of t-th bit in the truth table for two elements
//                        int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                        for (int a = 0; a <= 1; a++) {  // Traverse on a
//                            if (a == 0) {
//                                outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << X_unknown[q][t] << " " << -f[q][h] << " " << 0 << endl;
//                                mainClauseNum++;
//                            }
//                            if (a == 1) {
//                                outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -X_unknown[q][t] << " " << f[q][h] << " " << 0 << endl;
//                                mainClauseNum++;
//                            }
//                        }
//                    }
//                }
//            }
//
//            if ((j <= variableNumber + accuracy) && (k > variableNumber + accuracy)) { // If 1st element both is input variable or constant coefficient, the 2nd element is a gate.
//                for (int t = 0; t < pow(2, variableNumber + accuracy); t++) { // Traverse each bit in the truth table
//                    {
//                        int b = X_known[j - 1][t]; // Obtain the values of t-th bit in the truth table for 1st element
//                        int g = k - (variableNumber + accuracy + 1); // Calculate 2nd gate index in S_ij matrix
//                        for (auto c = 0; c <= 1; c++) {  // Traverse on c
//                            for (auto a = 0; a <= 1; a++) {  // Traverse on a
//                                if ((c == 0) && (a == 0)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile<< -S[q][j - 1] << " " << -S[q][k - 1] << " " << X_unknown[q][t] << " " << X_unknown[g][t] << " " << -f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//                                if ((c == 0) && (a == 1)) {
//                                    int h = Bin_To_Dec(b, c);
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -X_unknown[q][t] << " " << X_unknown[g][t] << " " << f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//                                if ((c == 1) && (a == 0)) {
//                                    int h = Bin_To_Dec(b, c);
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << X_unknown[q][t] << " " << -X_unknown[g][t] << " " << -f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//                                if ((c == 1) && (a == 1)) {
//                                    int h = Bin_To_Dec(b, c);
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[g][t] << " " << f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//
//            if ((j > variableNumber + accuracy) && (k > variableNumber + accuracy)) {
//                for (int t = 0; t < pow(2, variableNumber + accuracy); t++) {
//                    int x = j - (variableNumber + accuracy + 1);
//                    int y = k - (variableNumber + accuracy + 1);
//                    for (auto b = 0; b <= 1; b++) {
//                        for (auto c = 0; c <= 1; c++) {
//                            for (auto a = 0; a <= 1; a++) {
//                                if ((b == 0) && (c == 0) && (a == 0)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << X_unknown[q][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
//                                    " " << -f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                                if ((b == 0) && (c == 0) && (a == 1)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -X_unknown[q][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
//                                    " " << f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                                if ((b == 0) && (c == 1) && (a == 0)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << X_unknown[q][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
//                                    " " << -f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                                if ((b == 0) && (c == 1) && (a == 1)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -X_unknown[q][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
//                                    " " << f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                                if ((b == 1) && (c == 0) && (a == 0)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << X_unknown[q][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
//                                    " " << -f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                                if ((b == 1) && (c == 0) && (a == 1)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
//                                    " " << f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                                if ((b == 1) && (c == 1) && (a == 0)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << X_unknown[q][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
//                                    " " << -f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                                if ((b == 1) && (c == 1) && (a == 1)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
//                                    " " << f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//    cout << "The number of main clauses:" << mainClauseNum << endl;
//}

//void ConstructClause::ConstructMainClauseWithFence(ofstream& outputfile, vector < vector <int> > DAG){
//    int mainClauseNum = 0;
//
//    // Calculate the minIndex and maxIndex for s_ik of each gate
//    vector <int> minIndex;
//    vector <int> maxIndex;
//    for(auto i = 0; i < int(DAG.size()); i++){
//        if(i == 0){
//            for(auto p = 0; p < int(DAG[i].size()); p++){
//                minIndex.push_back(0);
//                maxIndex.push_back(variableNumber + accuracy - 1);
//            }
//        }
//        else{
//            for(auto p = 0; p < int(DAG[i].size()); p++){
//                minIndex.push_back(DAG[i-1][0] + variableNumber + accuracy);
//                maxIndex.push_back(DAG[i-1].back() + variableNumber + accuracy);
//            }
//        }
//    }
//
//    for(auto j = 0; j < gate; j++){
//        // Constraint on s_ik (at level t): have at least one fanin on level t−1
//        for(auto k = minIndex[j]; k <= maxIndex[j]; k++){
//            outputfile << S[j][k] << " ";
//        }
//        outputfile << 0 << endl;
//        mainClauseNum++;
//
//        // For gate i, gate k at the same level, S_ik = 0
//        for(auto k = 0; k < variableNumber+accuracy +j; k++){
//            if((k > maxIndex[j])){
//                outputfile << -S[j][k] << " " << 0 << endl;
//                mainClauseNum++;
//            }
//        }
//    }
//
//
//    for (auto i = variableNumber + accuracy + 1; i <= variableNumber + accuracy + gate; i++) {
//        int n0 = i - 1, j0 = N;
//        vector<vector<int> > InputComb;
//        vector<vector<int> > GateInputComb;
//        // Calculate C(i-1)_N, Mat stores different combinations in C(i-1)_N, where each combination is represented as a row
//
//        //cout << "The different input combinations for gate" << i << ":" << endl;
//        InputComb = combine(n0, j0);
//        cout << "The number of different input combinations for gate" << i << ": " << InputComb.size() << endl;
//
//        // If gate i (at level t) has two fains j and k (j < k), gate k should be on level t−1. Otherwise, delete this combination
//        int q = i - (variableNumber + accuracy + 1);  // The gate index in S_ij matrix and X_unknown matrix
//        for(auto x = 0; x < int(InputComb.size()); x++){
//            if((InputComb[x][1] >= minIndex[q] + 1) && (InputComb[x][1] <= maxIndex[q] + 1)){
//                GateInputComb.push_back(InputComb[x]);
//            }
//        }
//        cout << "The number of different input combinations for gate" << i << ": " << GateInputComb.size() << endl;
//
//
//        for (unsigned int p = 0; p < GateInputComb.size(); p++) { // Traverse each input combination
//            int j = GateInputComb[p][0], k = GateInputComb[p][1]; // Obtain each element in the input combination
//            if ((j <= variableNumber + accuracy) && (k <= variableNumber + accuracy)) {  // If the elements both are input variables or constant coefficients
//                for (int t = 0; t < (int)pow(2, variableNumber + accuracy); t++) {  // Traverse each bit in the truth table
//                    {
//                        int b = X_known[j - 1][t], c = X_known[k - 1][t]; // Obtain the values of t-th bit in the truth table for two elements
//                        int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                        for (int a = 0; a <= 1; a++) {  // Traverse on a
//                            if (a == 0) {
//                                outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << X_unknown[q][t] << " " << -f[q][h] << " " << 0 << endl;
//                                mainClauseNum++;
//                            }
//                            if (a == 1) {
//                                outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -X_unknown[q][t] << " " << f[q][h] << " " << 0 << endl;
//                                mainClauseNum++;
//                            }
//                        }
//                    }
//                }
//            }
//
//            if ((j <= variableNumber + accuracy) && (k > variableNumber + accuracy)) { // If 1st element both is input variable or constant coefficient, the 2nd element is a gate.
//                for (int t = 0; t < pow(2, variableNumber + accuracy); t++) { // Traverse each bit in the truth table
//                    {
//                        int b = X_known[j - 1][t]; // Obtain the values of t-th bit in the truth table for 1st element
//                        int g = k - (variableNumber + accuracy + 1); // Calculate 2nd gate index in S_ij matrix
//                        for (auto c = 0; c <= 1; c++) {  // Traverse on c
//                            for (auto a = 0; a <= 1; a++) {  // Traverse on a
//                                if ((c == 0) && (a == 0)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile<< -S[q][j - 1] << " " << -S[q][k - 1] << " " << X_unknown[q][t] << " " << X_unknown[g][t] << " " << -f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//                                if ((c == 0) && (a == 1)) {
//                                    int h = Bin_To_Dec(b, c);
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -X_unknown[q][t] << " " << X_unknown[g][t] << " " << f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//                                if ((c == 1) && (a == 0)) {
//                                    int h = Bin_To_Dec(b, c);
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << X_unknown[q][t] << " " << -X_unknown[g][t] << " " << -f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//                                if ((c == 1) && (a == 1)) {
//                                    int h = Bin_To_Dec(b, c);
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[g][t] << " " << f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//
//            if ((j > variableNumber + accuracy) && (k > variableNumber + accuracy)) {
//                for (int t = 0; t < pow(2, variableNumber + accuracy); t++) {
//                    int x = j - (variableNumber + accuracy + 1);
//                    int y = k - (variableNumber + accuracy + 1);
//                    for (auto b = 0; b <= 1; b++) {
//                        for (auto c = 0; c <= 1; c++) {
//                            for (auto a = 0; a <= 1; a++) {
//                                if ((b == 0) && (c == 0) && (a == 0)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << X_unknown[q][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
//                                    " " << -f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                                if ((b == 0) && (c == 0) && (a == 1)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -X_unknown[q][t] << " " << X_unknown[x][t] << " " << X_unknown[y][t] <<
//                                    " " << f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                                if ((b == 0) && (c == 1) && (a == 0)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << X_unknown[q][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
//                                    " " << -f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                                if ((b == 0) && (c == 1) && (a == 1)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -X_unknown[q][t] << " " << X_unknown[x][t] << " " << -X_unknown[y][t] <<
//                                    " " << f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                                if ((b == 1) && (c == 0) && (a == 0)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << X_unknown[q][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
//                                    " " << -f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                                if ((b == 1) && (c == 0) && (a == 1)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[x][t] << " " << X_unknown[y][t] <<
//                                    " " << f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                                if ((b == 1) && (c == 1) && (a == 0)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << X_unknown[q][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
//                                    " " << -f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                                if ((b == 1) && (c == 1) && (a == 1)) {
//                                    int h = Bin_To_Dec(b, c); // Convert binary values to a decimal value
//                                    outputfile << -S[q][j - 1] << " " << -S[q][k - 1] << " " << -X_unknown[q][t] << " " << -X_unknown[x][t] << " " << -X_unknown[y][t] <<
//                                    " " << f[q][h] << " " << 0 << endl;
//                                    mainClauseNum++;
//                                }
//
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//    cout << "The number of main clauses:" << mainClauseNum << endl;
//}

void ConstructClause::SymmetryBreakingConstraints(ofstream& outputfile){
    int symmetryConsNum = 0;

    // Symmetry breaking constraints: use all gates
    for(auto i = 0; i < gate - 1; i++){
        for(auto j = i+1; j < gate; j++){
            outputfile << S[j][i+variableNumber+accuracy] << " ";
        }
        outputfile << 0 << endl;
        symmetryConsNum++;
    }

    // Symmetry breaking constraints: co-lexicographically ordered steps & no reapplication of operands
    for (auto i = variableNumber + accuracy + 1; i < variableNumber + accuracy + gate; i++) {
        int n0 = i - 1, j0 = N;
        vector<vector<int> > GateInputComb0;
        //        vector<vector<int> > GateInputComb1;

        //cout << "The different input combinations for gate" << i << ":" << endl;
        GateInputComb0 = combine(n0, j0);
        //        GateInputComb1 = combine(n0+1, j0);

        cout << "The number of different input combinations for gate" << i << ": " << GateInputComb0.size() << endl;


        //        cout << "The number of different input combinations for gate" << i+1 << ": " << GateInputComb1.size() << endl;


        int q = i - (variableNumber + accuracy + 1);  // The gate index in S_ij matrix and X_unknown matrix

        //        // Symmetry breaking constraints: co-Lexicographically Ordered Steps, often increase running time
        //        for (unsigned int p = 0; p < GateInputComb0.size(); p++) { // Traverse each input combination
        //            int j0 = GateInputComb0[p][0], k0 = GateInputComb0[p][1]; // Obtain each element in the input combination
        //            for(auto m = 0; m < GateInputComb1.size(); m++){
        //                int j1 = GateInputComb1[m][0], k1 = GateInputComb1[m][1];
        //                if(k1 < k0){
        //                    outputfile << -S[q][j0-1] << " " << -S[q+1][j1-1] << " " << -S[q][k0-1] << " " << -S[q+1][k1-1] << " " << 0 << endl;
        //                    symmetryConsNum++;
        //                }
        //                else if((k0 == k1) && (j1 < j0)){
        //                    outputfile << -S[q][j0-1] << " " << -S[q+1][j1-1] << " " << -S[q][k0-1] << " " << -S[q+1][k1-1] << " " << 0 << endl;
        //                    symmetryConsNum++;
        //                }
        //            }
        //        }

        // Symmetry breaking constraints: no reapplication of operands
        for (unsigned int p = 0; p < GateInputComb0.size(); p++) { // Traverse each input combination
            int j0 = GateInputComb0[p][0], k0 = GateInputComb0[p][1]; // Obtain each element in the input combination

            for(auto x = q+1; x < gate; x++){
                outputfile << -S[q][j0-1] << " " << -S[q][k0-1] << " " << -S[x][j0-1] << " " << -S[x][q+variableNumber+accuracy] << " " << 0 << endl;
                symmetryConsNum++;
                outputfile << -S[q][j0-1] << " " << -S[q][k0-1] << " " << -S[x][k0-1] << " " << -S[x][q+variableNumber+accuracy] << " " << 0 << endl;
                symmetryConsNum++;
            }
        }
    }
    cout << "The number of symmetry breaking constraints:" << symmetryConsNum << endl;
}

void ConstructClause::MaxDepthConstraint(ofstream& outputfile, int maxFenceDepth, vector < vector <int> > DAG){
    int depthConstraintNum = 0;
    //     Constraint on s_ij, the connected two levels have limited depth.
    //     Once exceeding this level, s_ij =0

    for (auto x = maxFenceDepth; x < int(DAG.size()); x++){
        int limitDepth = x - maxFenceDepth;

        // The number of j whose S_ij=0, i.e., which exceeds max_depth.
        int zeroFaninNum = variableNumber + accuracy;
        for(auto l = 0; l < limitDepth; l++){
            zeroFaninNum = zeroFaninNum + int(DAG[l].size());
        }

        for(auto y = 0; y < int(DAG[x].size()); y++){
            int q = DAG[x][y];
            for(auto p = 0; p < zeroFaninNum; p++){
                outputfile << -S[q][p] << " " << 0 << endl;
                depthConstraintNum++;
            }
        }
    }
    cout << "The number of maxDepth constraints:" << depthConstraintNum << endl;
}

void ConstructClause::MaxFanout(ofstream& outputfile, int maxFanout){
    int fanoutConstraintNum = 0;
    for(auto i = 0; i < variableNumber+accuracy+gate-1; i++){
        // For the PI, the number of fanout cannot exceed maxFanout.
        if(i < variableNumber+accuracy){
            vector<vector<int> > combAtMostMaxFanout;
            combAtMostMaxFanout = combine(gate, maxFanout+1);

            for(auto p = 0; p < int(combAtMostMaxFanout.size()); p++){
                for(auto q = 0; q < int(combAtMostMaxFanout[p].size()); q++){
                    outputfile << -S[combAtMostMaxFanout[p][q]-1][i] << " ";
                }
                outputfile << 0 << endl;
                fanoutConstraintNum++;
            }
        }
        //For the gates, the number of fanout cannot exceed maxFanout.
        else{
            vector<vector<int> > combAtMostMaxFanout1;
            combAtMostMaxFanout1 = combine((variableNumber+accuracy+gate-1)-i, maxFanout+1);

            for(auto p = 0; p < int(combAtMostMaxFanout1.size()); p++){
                for(auto q = 0; q < int(combAtMostMaxFanout1[p].size()); q++){
                    outputfile << -S[(i-variableNumber-accuracy)+combAtMostMaxFanout1[p][q]][i] << " ";
                }
                outputfile << 0 << endl;
                fanoutConstraintNum++;
            }

        }
    }

    cout << "The number of maxFanout constraints:" << fanoutConstraintNum << endl;

}


// Construct 1st contraint clause: the operator has 2-input fanins.
void ConstructClause::Construct1stClause(ofstream& outputfile, context& c, solver& s) {
    int firststClauseNum = 0;
//    for(auto i = 0; i < gate; i++){
//    	expr tmp = c.int_const("tmp");
//		tmp = c.int_val(0);
//    	for(auto j = 0; j < S[i].size(); j++){
//    		cout << S_Z3[i][j] << endl;
//    		tmp = tmp + int(S_Z3[i][j]);
//    	}
//    	s.add(tmp == 2);
//    }
//    s.add((S_Z3[0][0]) + int(S_Z3[0][1]) + int(S_Z3[0][2]) + int(S_Z3[0][3]) + int(S_Z3[0][4]) == 2);
//    s.add(S_Z3[1][0] + S_Z3[1][1] + S_Z3[1][2]+ S_Z3[1][3]+ S_Z3[1][4] + S_Z3[1][5] == 2);
//    s.add(S_Z3[2][0] + S_Z3[2][1] + S_Z3[2][2]+ S_Z3[2][3]+ S_Z3[2][4] + S_Z3[2][5] + S_Z3[2][6] == 2);
//    s.add(S_Z3[3][0] + S_Z3[3][1] + S_Z3[3][2]+ S_Z3[3][3]+ S_Z3[3][4] + S_Z3[3][5] + S_Z3[3][6]== 2);

    for (int i = variableNumber + accuracy + 1; i <= variableNumber + accuracy + gate; i++) {
            int q = i - (variableNumber + accuracy + 1);  // The gate index in the S_ij matrix
            int n1 = i - 1;
            if (n1 == N) {
                for (unsigned int l = 0; l < S[q].size(); l++) {
                    outputfile << S[q][l] << " " << 0 << endl;
                    s.add(S_Z3[q][l]);
                    firststClauseNum++;
                }
            }
            else {
                int j1 = N + 1;
                vector<vector<int> > combAtMost4;
                // Calculate C(i-1)_(N+1), Mat1 stores different combinations in C(i-1)_(N+1), where each combination is represented as a row
                combAtMost4 = combine(n1, j1);


                //cout << "The different combinations for C(i-1)_3 in combAtMost2 for gate" << i << ":" << endl;
                //            int combNumAtMost2 = 0;
                //            for (unsigned int p = 0; p < combAtMost2.size(); p++) {
                //                for (unsigned int q = 0; q < combAtMost2[p].size(); q++) {
                //                    //combAtMost2[p][q] -= 1;
                //                    cout << combAtMost2[p][q] << " ";
                //                }
                //                cout << endl;
                //                combNumAtMost2++;
                //            }



//                cout << "The number of different combinations for C(i-1)_5 in combAtMost4 for gate" << i << ": " << combAtMost4.size() << endl;

                // Ensures that for any gate 𝑖, there are at most 4 inputs.
                for (unsigned int p = 0; p < combAtMost4.size(); p++) {  // Traverse on each input combination
                	expr tmp = c.bool_const("tmp");
                	tmp = c.bool_val(false);
                    for (unsigned int l = 0; l < combAtMost4[p].size(); l++) {
                        outputfile << -S[q][combAtMost4[p][l]-1] << " ";
                        tmp = tmp || (!S_Z3[q][combAtMost4[p][l]-1]);
                    }
                    outputfile << 0 << endl;
                    s.add(tmp);
                    firststClauseNum++;
                }

                int n2 = i - 1, j2 = i - 1 - (N - 1);
                vector<vector<int> > combAtLeast4;
                // Calculate C(i-1)_(i-2), Mat2 stores different combinations in C(i-1)_(i-2), where each combination is represented as a row
                combAtLeast4 = combine(n2, j2);


                //cout << "The different combinations for C(i-1)_(i-2) in combAtLeast2 for gate" << i << ":" << endl;
                //            int combNumAtLeast2 = 0;
                //            for (unsigned int p = 0; p < combAtLeast2.size(); p++) {
                //                for (unsigned int q = 0; q < combAtLeast2[p].size(); q++) {
                //                    //combAtLeast2[p][q] -= 1;
                //                    cout << combAtLeast2[p][q] << " ";
                //                }
                //                cout << endl;
                //                combNumAtLeast2++;
                //            }


//                cout << "The number of different combinations for C(i-1)_(i-4) in combAtLeast2 for gate" << i << ": " << combAtLeast4.size() << endl;

                // Ensures that for any gate 𝑖, there are at least 4 inputs.
                for (unsigned int p = 0; p < combAtLeast4.size(); p++) {
                	expr tmp = c.bool_const("tmp");
                	tmp = c.bool_val(false);
                    for (unsigned int j = 0; j < combAtLeast4[0].size(); j++) {
                        int t = combAtLeast4[p][j]-1;
                        outputfile << S[q][t] << " ";
                        tmp = tmp || (S_Z3[q][t]);
                    }
                    outputfile << 0 << endl;
                    s.add(tmp);
                    firststClauseNum++;
                }

            }
        }
//    cout << "The number of first clauses:" << firststClauseNum << endl;
}


MatInfo ConstructClause::PrepareForConstruct2ndClause(){
    //    vector<int> Sum((int)pow(2, variableNumber), 0);
    int k = 0;
    // Calculate the sum of each bit in the truth table for input variables. E.g., Sum:0 1 1 2
    for (auto t = 0; t < (int)(pow(2, variableNumber + accuracy)); t = t + (int)pow(2, accuracy)) {
        for (auto i = 0; i < variableNumber; i++) {
            SumColumn[k] = SumColumn[k] + X_known[i][t];
        }
        k++;
    }
//    cout << "The sum of each n input variables: ";
    for (unsigned int i = 0; i < SumColumn.size(); i++) {
//        cout << SumColumn[i] << " ";
    }
//    cout << endl;


    // Calculate the number of combinations for input variables in each G(i). E.g., Index:1 2 1
    vector <int> XnNumInG(variableNumber + 1);
    for (int i = 0; i < variableNumber + 1; i++) {
        for (unsigned int j = 0; j < SumColumn.size(); j++) {
            if (SumColumn[j] == i)
                XnNumInG[i]++;
        }
    }

//    cout << "The number of input variables combinations in the feature vector G: ";
    for (unsigned int i = 0; i < XnNumInG.size(); i++) {
//        cout << XnNumInG[i] << " ";
    }
//    cout << endl;

    // Calculate the number of combinations for input variables and constant coefficients in each G(i). E.g., Number:2 4 2
    vector <int> Xn_mNumInG(variableNumber + 1);
//    cout << "The number of combinations for input variables and constant coefficients in the feature vector G: ";
    for (unsigned int i = 0; i < Xn_mNumInG.size(); i++) {
        Xn_mNumInG[i] = XnNumInG[i] * (int)pow(2, accuracy);
//        cout << Xn_mNumInG[i] << " ";
    }
//    cout << endl;


    // Store each combination of input variables for each G(i). E.g., Mat3: [0; 1,2; 3]
    vector < vector <int> > indexOfXnInG(variableNumber + 1, vector <int>(0));
    for (unsigned int i = 0; i < featureVector.size(); i++) {
        for (unsigned int k = 0; k < SumColumn.size(); k++) {
            if (SumColumn[k] == i) {
                indexOfXnInG[i].push_back(k);
            }
        }
    }

//    cout << "The index of input variables X(n) in FV: " << endl;
    for (unsigned int i = 0; i < indexOfXnInG.size(); i++) {
        for (unsigned int j = 0; j < indexOfXnInG[i].size(); j++) {
//            cout << indexOfXnInG[i][j] << " ";
        }
//        cout << endl;
    }

    // Store each combination of input variables and constant coefficients for each G(i). E.g., Mat4:[0 1; 2 3 4 5; 6 7]
    vector < vector <int> > indexOfXn_mInG(variableNumber + 1, vector <int>(0));
    for (unsigned int i = 0; i < indexOfXn_mInG.size(); i++) {
        indexOfXn_mInG[i].resize(Xn_mNumInG[i]);
    }

    for (unsigned int s1 = 0; s1 < indexOfXnInG.size(); s1++) {
        for (unsigned int s2 = 0; s2 < indexOfXnInG[s1].size(); s2++) {
            for (int s3 = 0; s3 < pow(2, accuracy); s3++) {
                indexOfXn_mInG[s1][(int)pow(2, accuracy) * s2 + s3] = (int)pow(2, accuracy) * indexOfXnInG[s1][s2] + s3;
                //cout << Mat4[s1][2*s2+s3] << " ";
            }

        }
    }

//    cout << "The index of input variables and constant coefficients X(n+m) in FV: " << endl;
    for (unsigned int i = 0; i < indexOfXn_mInG.size(); i++) {
        for (unsigned int j = 0; j < indexOfXn_mInG[i].size(); j++) {
//            cout << indexOfXn_mInG[i][j] << " ";
        }
//        cout << endl;
    }

    MatInfo IndexMatrix(indexOfXnInG, indexOfXn_mInG);
    return IndexMatrix;
}

void ConstructClause::ConstructFVClauseWithoutESPRESSO(string prefix, ofstream& outputfile, vector <int> featureVector, vector <int> largestCubeVector, vector<int> toBeAssignedCV1, vector <vector <IndexRelation> > AllVarRelation, int coarseGrain1, MatInfo IndexMatrix, vector <int> toBeAssignedCV2, int coarseGrain2, context& c, solver& s){

    int FVClauseNum = 0;

    // Determine the index in G for each Y-variable
    vector < vector <int> > FVIndexofY (variableNumber+1, vector <int> {});
    for(auto p = 0; p < int(FVIndexofY.size()); p++){
        for(auto q = 0; q < int(AllVarRelation[1].size()); q++){
            if(AllVarRelation[1][q].first[1] == p){
                FVIndexofY[p].push_back(AllVarRelation[1][q].first[0]);
            }
        }
    }

    // Determine the index in G for each Z-variable
    vector < vector <int> > FVIndexofZ (variableNumber+1, vector <int> {});
    for(auto p = 0; p < int(FVIndexofZ.size()); p++){
        for(auto q = 0; q < int(AllVarRelation[0].size()); q++){
            if(AllVarRelation[0][q].first[1] == p){
                FVIndexofZ[p].push_back(AllVarRelation[0][q].first[0]);
            }
        }
    }

    // Determine the internal RFV, update Remaining FV after inserting the cubes with largest CV in the truth table
    vector <int> InternalRFV(variableNumber+1);

    for(auto i = 0; i < int(featureVector.size()); i++){
        InternalRFV[i] = featureVector[i] - largestCubeVector[i];

        // If featureVector[i] is equal to the maxFV[i], all values equal to 1.
        if(featureVector[i] == int((IndexMatrix.second)[i].size())){
            InternalRFV[i] = 0;
            //            for(auto m = 0; m < FVIndexofY[i].size(); m++){
            //                int index = FVIndexofY[i][m] - YVariable[0];
            //                outputfile << YVariable[index] << " " << 0 << endl;
            //                FVClauseNum++;
            //            }
            for(auto j = 0; j < int((IndexMatrix.first)[i].size()); j++){
                for(auto k = 0; k < int(AssMatrix.size()); k++){
                	AssMatrix[k][IndexMatrix.first[i][j]] = 1;
                }
            }
        }

        // If Internal RemainFV[i] is 0, set elements (i.e., value hasn't been determined) in the corresponding columns(i.e., G[i]) to 0.
        else if(InternalRFV[i] == 0 && featureVector[i] != int((IndexMatrix.second)[i].size())){
            for(auto j = 0; j < int((IndexMatrix.first)[i].size());j++){  // Tranverse each column
                for(auto k = 0; k < int(AssMatrix.size()); k++){
                    if(AssMatrix[k][IndexMatrix.first[i][j]] == -1){
                    	AssMatrix[k][IndexMatrix.first[i][j]] = 0;
                    }
                }
            }
        }

    }

//    cout << "Assignment matrix after inserting the cube with largest cube vector:" << endl;
//    for(auto i = 0; i < int(AssMatrix.size()); i++){
//        for(auto j = 0; j < int(AssMatrix[i].size()); j++){
//            cout << AssMatrix[i][j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl << endl;


    // For the elemets equal to 1 in the assignment matrix, delete corresponding Z-variables in FVIndexZ.
    for(auto p = 0; p < int(AssMatrix[0].size()); p++){
        for(auto q = 0; q < int(AssMatrix.size()); q=q+coarseGrain1){
            //            if(AssMat[q][p] == 0){
            //                int l = (p * int(pow(2, accuracy)) + q)/coarseGrain1;
            //                outputfile << -ZVariable[l] << " " << 0 << endl;
            //                FVClauseNum++;
            //            }

            //            if(AssMat[q][p] == 1){
            //                int l = (p * int(pow(2, accuracy)) + q)/coarseGrain1;
            //                outputfile << ZVariable[l] << " " << 0 << endl;
            //                FVClauseNum++;
            //            }

            int flag = 1;
            for(auto m = 0; m < coarseGrain1; m++){
                if(AssMatrix[q+m][p] != 1){
                    flag = 0;
                    break;
                }
            }

            if(flag){
                int l = (p * int(pow(2, accuracy)) + q)/coarseGrain1 + ZVariable[0];
                int loc = SumColumn[p];
                auto iter = find(FVIndexofZ[loc].begin(), FVIndexofZ[loc].end(), l);
                FVIndexofZ[loc].erase(iter);
            }

        }
    }

    // For the elemets equal to 1 in the assignment matrix, delete corresponding Y-variables in FVIndexY.
    for(auto p = 0; p < int(AssMatrix[0].size()); p++){
        for(auto q = 0; q < int(AssMatrix.size()); q=q+coarseGrain2){
            //                if(AssMat[q][p] == 0){
            //                    int l = (p * int(pow(2, accuracy)) + q)/coarseGrain2;
            //                    outputfile << -YVariable[l] << " " << 0 << endl;
            //                    FVClauseNum++;
            //                }

            //                if(AssMat[q][p] == 1){
            //                    int l = (p * int(pow(2, accuracy)) + q)/coarseGrain2;
            //                    outputfile << YVariable[l] << " " << 0 << endl;
            //                    FVClauseNum++;
            //                }

            int flag = 1;
            for(auto m = 0; m < coarseGrain2; m++){
                if(AssMatrix[q+m][p] != 1){
                    flag = 0;
                    break;
                }
            }

            if(flag){
                int l = (p * int(pow(2, accuracy)) + q)/coarseGrain2 + YVariable[0];
                int loc = SumColumn[p];
                auto iter = find(FVIndexofY[loc].begin(), FVIndexofY[loc].end(), l);
                FVIndexofY[loc].erase(iter);
            }

        }
    }

    // Determined Cube Vector (DCV): For the element 0/1 in the Assignment matrix, construct CNF clauses for corresponding X-variables
    for(auto p = 0; p < int(AssMatrix[0].size()); p++){
        for(auto q = 0; q < int(AssMatrix.size()); q++){
            if(AssMatrix[q][p] == 0){
                int index = p * int(pow(2, accuracy)) + q;
                outputfile << -X_unknown[gate-1][index] << " " << 0 << endl;
                s.add(!X_Unknown_Z3[gate-1][index]);
                FVClauseNum++;
            }

            else if(AssMatrix[q][p] == 1){
                int index = p * int(pow(2, accuracy)) + q;
                outputfile << X_unknown[gate-1][index] << " " << 0 << endl;
                s.add(X_Unknown_Z3[gate-1][index]);
                FVClauseNum++;

                // Delete corresponding X-variables in IndexMatrix.second
                int loc = SumColumn[p];
                auto iter = find(IndexMatrix.second[loc].begin(),IndexMatrix.second[loc].end(),index);
                IndexMatrix.second[loc].erase(iter);
            }

        }
    }

    // Determine the final remaining feature vector.(fine-grained search)
    vector <int> RemainFV(variableNumber+1);
    for(auto k = 0; k < int(RemainFV.size()); k++){
        RemainFV[k] = featureVector[k] - largestCubeVector[k] - toBeAssignedCV1[k]- toBeAssignedCV2[k];
    }

    //    // The relationship between Y variables and Z variables
    //    for(auto t = 0; t < AllVarRelation[0].size();t++){
    //        //        outputfile << AllVarRelation[0][t].first[0] << " ";
    //        //        for(auto l = 0; l < (coarseGrain1/coarseGrain2); l++){
    //        //            outputfile << -AllVarRelation[0][t].second[l] << " ";
    //        //        }
    //        //        outputfile << 0 << endl;
    //        //        FVClauseNum++;
    //
    //        for(auto l = 0; l < (coarseGrain1/coarseGrain2); l++){
    //            outputfile << -AllVarRelation[0][t].first[0] << " " << AllVarRelation[0][t].second[l] << " " << 0 << endl;
    //            FVClauseNum++;
    //        }
    //
    //    }
    //
    //    // The relationship between X variables and Y variables
    //    for(auto t = 0; t < AllVarRelation[1].size();t++){
    //        //        outputfile << AllVarRelation[1][t].first[0] << " ";
    //        //        for(auto l = 0; l < coarseGrain2; l++){
    //        //            outputfile << -AllVarRelation[1][t].second[l] << " ";
    //        //        }
    //        //        outputfile << 0 << endl;
    //        //        FVClauseNum++;
    //
    //        for(auto l = 0; l < coarseGrain2; l++){
    //            outputfile << -AllVarRelation[1][t].first[0] << " " << AllVarRelation[1][t].second[l] << " " << 0 << endl;
    //            FVClauseNum++;
    //        }
    //
    //    }

    // Formulate CNF clauses to satisfy RFV[i]
    for(auto i = 0; i < int(featureVector.size()); i++){
        if(InternalRFV[i]!=0){
//            cout << "For G[" << i << "]:" << endl;

            // The total number of Z-vars in G[i]
            int ZSumNum = int(FVIndexofZ[i].size());
            // The number of Z-vars for which need to assign 1.
            int ZNum = toBeAssignedCV1[i]/coarseGrain1;

            // The total number of Y-vars in G[i]
            int YSumNum = int(FVIndexofY[i].size());
            // The number of Y-vars for which need to assign 1.
            int YNum = toBeAssignedCV2[i]/coarseGrain2;

            // If there doesn't exist coarse granularities, construct CNF clauses for the X-variables
            if(ZNum == 0){
//                cout << "ZNum=0" << endl;
                if(YNum == 0){
//                    cout << "YNum=0" << endl;
                    //    for(auto x = 0; x < int(FVIndexofY[i].size()); x++){
                    //        outputfile << -FVIndexofY[i][x] << " " << 0 << endl;
                    //        FVClauseNum++;
                    //    }
//                    cout << "Remain FV= " << RemainFV[i] << endl;
                    int n0 = int(IndexMatrix.second[i].size()), j0 = n0 - (RemainFV[i] - 1);
                    vector<vector<int> > combAtLeastGi;
                    // Calculate Cn_(n-(G[i-1)), Mat6 stores different combinations in Cn_(n-(G[i]-1)), where each combination is represented as a row
                    combAtLeastGi = combine(n0, j0);  //which is index in indexOfXn_mInG

//                    cout << "The number of different combinations combAtLeastXi for G[" << i << "]: " << combAtLeastGi.size() << endl;

                    // Ensures that for any gate 𝑖, there are at least G[i] inputs.
                    for (unsigned int p = 0; p < combAtLeastGi.size(); p++) {
                    	expr tmp = c.bool_const("tmp");
                    	tmp = c.bool_val(false);
                        for (unsigned int q = 0; q < combAtLeastGi[p].size(); q++) {
                            int x = combAtLeastGi[p][q]-1;
                            int l = IndexMatrix.second[i][x];
                            outputfile << X_unknown[gate - 1][l] << " ";
                            tmp = tmp || X_Unknown_Z3[gate - 1][l];
                        }
                        outputfile << 0 << endl;
                        s.add(tmp);
                        FVClauseNum++;
                    }

                    // Ensures that for any gate 𝑖, there are at most G[i] inputs.
                    int n1 = int(IndexMatrix.second[i].size()), j1 = RemainFV[i] + 1;
                    vector<vector<int> > combAtMostGi;
                    // Calculate Cn_(G[i]+1), Mat7 stores different combinations in Cn_(G[i]+1), where each combination is represented as a row.
                    combAtMostGi = combine(n1, j1);

//                    cout << "The number of different combinations in combAtMostXi for G[" << i << "]: " << combAtMostGi.size() << endl;

                    // Ensures that for any gate 𝑖, there are at most G[i] inputs.
                    for (unsigned int p = 0; p < combAtMostGi.size(); p++) {
                    	expr tmp = c.bool_const("tmp");
                    	tmp = c.bool_val(false);
                        for (unsigned int q = 0; q < combAtMostGi[p].size(); q++) {
                            int x = combAtMostGi[p][q]-1;
                            int l = IndexMatrix.second[i][x];
                            outputfile << -X_unknown[gate - 1][l] << " ";
                            tmp = tmp || (!X_Unknown_Z3[gate - 1][l]);
                        }
                        outputfile << 0 << endl;
                        s.add(tmp);
                        FVClauseNum++;
                    }
                }

                else if(YNum != 0){
//                    cout << "YNum=" << YNum << endl;
//                    cout << "Remain FV=" << RemainFV[i] << endl;

                    // The relationship between X variables and Y variables
                    for(auto x = 0; x < int(FVIndexofY[i].size()); x++){
                        int t = FVIndexofY[i][x] - YVariable[0];
                        for(auto l = 0; l < coarseGrain2; l++){
                        	int m = AllVarRelation[1][t].first[0] - YVariable[0];
						    int n = AllVarRelation[1][t].second[l] - X_unknown[gate - 1][0];

                            outputfile << -AllVarRelation[1][t].first[0] << " " << AllVarRelation[1][t].second[l] << " " << 0 << endl;
                            s.add(!Y_Variable_Z3[m] || X_Unknown_Z3[gate-1][n]);
                            FVClauseNum++;
                        }
                    }

                    // Coarse granularity constraint for Y-variables
                    int n2 = YSumNum, j2 = n2 - (YNum - 1);
                    vector<vector<int> > combAtLeastYi;
                    // Calculate Cn_(n-(G[i-1)), Mat6 stores different combinations in Cn_(n-(G[i]-1)), where each combination is represented as a row
                    combAtLeastYi = combine(n2, j2);  //which is index in indexOfXn_mInG
//                    cout << "The number of different combinations in combAtLeastYi for G[" << i << "]: " << combAtLeastYi.size() << endl;

                    for (unsigned int p = 0; p < combAtLeastYi.size(); p++) {
                    	expr tmp = c.bool_const("tmp");
                    	tmp = c.bool_val(false);
                        for (unsigned int q = 0; q < combAtLeastYi[p].size(); q++) {
                            int x = combAtLeastYi[p][q]-1;
                            outputfile << FVIndexofY[i][x] << " ";

                            int index = FVIndexofY[i][x] - YVariable[0];
                            tmp = tmp || Y_Variable_Z3[index];
                        }
                        outputfile << 0 << endl;
                        s.add(tmp);
                        FVClauseNum++;
                    }

                    // Ensures that for any gate 𝑖, there are at most G[i] inputs.
                    int n3 = YSumNum, j3 = YNum + 1;
                    vector<vector<int> > combAtMostYi;
                    // Calculate Cn_(G[i]+1), Mat7 stores different combinations in Cn_(G[i]+1), where each combination is represented as a row.
                    combAtMostYi = combine(n3, j3);
//                    cout << "The number of different combinations in combAtMostYi for G[" << i << "]: " << combAtMostYi.size() << endl;

                    for (unsigned int p = 0; p < combAtMostYi.size(); p++) {
                    	expr tmp = c.bool_const("tmp");
                    	tmp = c.bool_val(false);
                        for (unsigned int q = 0; q < combAtMostYi[p].size(); q++) {
                            int x = combAtMostYi[p][q]-1;
                            outputfile << -FVIndexofY[i][x] << " ";

                            int index = FVIndexofY[i][x] - YVariable[0];
                            tmp = tmp || (!Y_Variable_Z3[index]);
                        }
                        outputfile << 0 << endl;
                        s.add(tmp);
                        FVClauseNum++;
                    }

                    // Fine granularity constraint for X-variables

                    // Calculate the one-hot combination of Y-variables
                    vector< vector<int> > combCoarseSearchY = ProduceBinComb(YSumNum, YNum);

//                    cout << "The number of different combinations for combYi: " << combCoarseSearchY.size() << endl;

                    // Calculate the number of X-variables whose values have not been determined
                    int XSumNum = (YSumNum-YNum)*coarseGrain2;
                    //                    int XSumNum = int(IndexMatrix.second[i].size()) - YNum*coarseGrain2;

                    int n4 = XSumNum, j4 = n4 - (RemainFV[i] - 1);
                    vector<vector<int> > combAtLeastXi;
                    // Calculate Cn_(n-(G[i-1)), Mat6 stores different combinations in Cn_(n-(G[i]-1)), where each combination is represented as a row
                    combAtLeastXi = combine(n4, j4);  //which is index in indexOfXn_mInG


//                    cout << "The number of different combinations in combAtLeastXi for G[" << i << "]: " << combAtLeastXi.size() << endl;

                    // Ensures that for any gate 𝑖, there are at most G[i] inputs.
                    int n5 = XSumNum, j5 = RemainFV[i] + 1;
                    vector<vector<int> > combAtMostXi;
                    // Calculate Cn_(G[i]+1), Mat7 stores different combinations in Cn_(G[i]+1), where each combination is represented as a row.
                    combAtMostXi = combine(n5, j5);


//                    cout << "The number of different combinations in combAtMostXi for G[" << i << "]: " << combAtMostXi.size() << endl;

                    for(auto j = 0; j < int(combCoarseSearchY.size()); j++){
                        vector <int> YVar{};
                        vector <int> XVar{};
                        for(auto k = 0; k < int(combCoarseSearchY[j].size()); k++){
                            if(combCoarseSearchY[j][k] == 0){
                                YVar.push_back(FVIndexofY[i][k]);
                                int l = FVIndexofY[i][k]-YVariable[0];
                                for(int x = 0; x < coarseGrain2; x++){
                                    XVar.push_back(AllVarRelation[1][l].second[x]);
                                }
                            }
                            else if(combCoarseSearchY[j][k] == 1){
                                YVar.push_back(-FVIndexofY[i][k]);
                            }

                        }

                        //    if(RemainFV[i] == 0){
                        //        for(auto l = 0; l < int(YVar.size()); l++){
                        //            outputfile << YVar[l] << " ";
                        //        }

                        //        for(auto x = 0; x < int(XVar.size()); x++){
                        //            outputfile << -XVar[x] << " ";
                        //        }
                        //        outputfile << 0 << endl;
                        //        FVClauseNum++;
                        //    }

                        //    else if(RemainFV[i] != 0){
                        for(auto m = 0; m < int(combAtLeastXi.size()); m++){
                        	expr tmp = c.bool_const("tmp");
                        	tmp = c.bool_val(false);
                            for(auto l = 0; l < int(YVar.size()); l++){
                                outputfile << YVar[l] << " ";

                                if(YVar[l] > 0){
                                	int index = YVar[l] - YVariable[0];
                                	tmp = tmp || (Y_Variable_Z3[index]);
                                }
                                else if(YVar[l] < 0){
                                	int index = -YVar[l] - YVariable[0];
                                	tmp = tmp || (!Y_Variable_Z3[index]);
                                }
                            }

                            for(auto n = 0; n < int(combAtLeastXi[m].size()); n++){
                            	int l = XVar[combAtLeastXi[m][n]-1];
                                int index = XVar[combAtLeastXi[m][n]-1] - X_unknown[gate - 1][0];
                                outputfile << l << " ";

                                tmp = tmp || (X_Unknown_Z3[gate-1][index]);
                            }
                            outputfile << 0 << endl;

                            s.add(tmp);
                            FVClauseNum++;
                        }

                        for (unsigned int p = 0; p < combAtMostXi.size(); p++) {
                        	expr tmp = c.bool_const("tmp");
                        	tmp = c.bool_val(false);
                            for(auto l = 0; l < int(YVar.size()); l++){
                                outputfile << YVar[l] << " ";

                                if(YVar[l] > 0){
                                	int index = YVar[l] - YVariable[0];
                                	tmp = tmp || (Y_Variable_Z3[index]);
                                }
                                else if(YVar[l] < 0){
                                	int index = -YVar[l] - YVariable[0];
                                	tmp = tmp || (!Y_Variable_Z3[index]);
                                }

                            }
                            for (unsigned int q = 0; q < combAtMostXi[p].size(); q++) {
                                int l = XVar[combAtMostXi[p][q]-1];
                                outputfile << -l << " ";

                                int index = XVar[combAtMostXi[p][q]-1] - X_unknown[gate - 1][0];
                                tmp = tmp || (!X_Unknown_Z3[gate-1][index]);
                            }
                            outputfile << 0 << endl;

                            s.add(tmp);
                            FVClauseNum++;
                        }
                        // }
                    }
                }
            }

            // If there exist coarse granularities
            else if(ZNum != 0){
//                cout << "ZNum=" << ZNum << endl;

                if(YNum == 0){
//                    cout << "YNum=0" << endl;
//                    cout << "Remain FV=" << RemainFV[i] << endl;

                    // The relationship between X variables and Z variables
                    for(auto x = 0; x < int(FVIndexofZ[i].size()); x++){
                        int t = FVIndexofZ[i][x] - ZVariable[0];
                        for(auto l = 0; l < coarseGrain1; l++){
                        	int m = AllVarRelation[2][t].first[0] - ZVariable[0];
                        	int n = AllVarRelation[2][t].second[l] - X_unknown[gate - 1][0];

                            outputfile << -AllVarRelation[2][t].first[0] << " " << AllVarRelation[2][t].second[l] << " " << 0 << endl;
                            s.add(!Z_Variable_Z3[m] || X_Unknown_Z3[gate-1][n]);
                            FVClauseNum++;
                        }
                    }

                    // Coarse granularity constraint for Z-variables
                    int n2 = ZSumNum, j2 = n2 - (ZNum - 1);
                    vector<vector<int> > combAtLeastZi;
                    // Calculate Cn_(n-(G[i-1)), Mat6 stores different combinations in Cn_(n-(G[i]-1)), where each combination is represented as a row
                    combAtLeastZi = combine(n2, j2);  //which is index in indexOfXn_mInG
//                    cout << "The number of different combinations in combAtLeastZi for G[" << i << "]: " << combAtLeastZi.size() << endl;

                    for (unsigned int p = 0; p < combAtLeastZi.size(); p++) {
                    	expr tmp = c.bool_const("tmp");
                    	tmp = c.bool_val(false);
                        for (unsigned int q = 0; q < combAtLeastZi[p].size(); q++) {
                            int x = combAtLeastZi[p][q]-1;
                            outputfile << FVIndexofZ[i][x] << " ";

                            int index = FVIndexofZ[i][x] - ZVariable[0];
                            tmp = tmp || Z_Variable_Z3[index];
                        }
                        outputfile << 0 << endl;
                        s.add(tmp);
                        FVClauseNum++;
                    }

                    // Ensures that for any gate 𝑖, there are at most G[i] inputs.
                    int n3 = ZSumNum, j3 = ZNum + 1;
                    vector<vector<int> > combAtMostZi;
                    // Calculate Cn_(G[i]+1), Mat7 stores different combinations in Cn_(G[i]+1), where each combination is represented as a row.
                    combAtMostZi = combine(n3, j3);
//                    cout << "The number of different combinations in combAtMostZi for G[" << i << "]: " << combAtMostZi.size() << endl;

                    for (unsigned int p = 0; p < combAtMostZi.size(); p++) {
                    	expr tmp = c.bool_const("tmp");
                    	tmp = c.bool_val(false);
                        for (unsigned int q = 0; q < combAtMostZi[p].size(); q++) {
                            int x = combAtMostZi[p][q]-1;
                            outputfile << -FVIndexofZ[i][x] << " ";

                            int index = FVIndexofZ[i][x] - ZVariable[0];
                            tmp = tmp || (!Z_Variable_Z3[index]);
                        }
                        outputfile << 0 << endl;
                        s.add(tmp);
                        FVClauseNum++;
                    }

                    // Fine granularity constraint for X-variables
                    vector< vector<int> > combCoarseSearchZ = ProduceBinComb(ZSumNum, ZNum);
//                    cout << "The number of different combinations for combZi: " << combCoarseSearchZ.size() << endl;

                    // Calculate the number of X-variables whose values have not been determined
                    int XSumNum = (ZSumNum-ZNum)*coarseGrain1;

                    int n4 = XSumNum, j4 = n4 - (RemainFV[i] - 1);
                    vector<vector<int> > combAtLeastXi;
                    // Calculate Cn_(n-(G[i-1)), Mat6 stores different combinations in Cn_(n-(G[i]-1)), where each combination is represented as a row
                    combAtLeastXi = combine(n4, j4);  //which is index in indexOfXn_mInG


//                    cout << "The number of different combinations in combAtLeastXi for G[" << i << "]: " << combAtLeastXi.size() << endl;

                    // Ensures that for any gate 𝑖, there are at most G[i] inputs.
                    int n5 = XSumNum, j5 = RemainFV[i] + 1;
                    vector<vector<int> > combAtMostXi;
                    // Calculate Cn_(G[i]+1), Mat7 stores different combinations in Cn_(G[i]+1), where each combination is represented as a row.
                    combAtMostXi = combine(n5, j5);


//                    cout << "The number of different combinations in combAtMostXi for G[" << i << "]: " << combAtMostXi.size() << endl;

                    for(auto j = 0; j < int(combCoarseSearchZ.size()); j++){
                        vector <int> ZVar{};
                        vector <int> XVar{};
                        for(auto k = 0; k < int(combCoarseSearchZ[j].size()); k++){
                            if(combCoarseSearchZ[j][k] == 0){
                                ZVar.push_back(FVIndexofZ[i][k]);
                                int l = FVIndexofZ[i][k]-ZVariable[0];
                                for(int x = 0; x < coarseGrain1; x++){
                                    XVar.push_back(AllVarRelation[2][l].second[x]);
                                }
                            }
                            else if(combCoarseSearchZ[j][k] == 1){
                                ZVar.push_back(-FVIndexofZ[i][k]);
                            }

                        }

                        //    if(RemainFV[i] == 0){
                        //        for(auto l = 0; l < int(YVar.size()); l++){
                        //            outputfile << YVar[l] << " ";
                        //        }

                        //        for(auto x = 0; x < int(XVar.size()); x++){
                        //            outputfile << -XVar[x] << " ";
                        //        }
                        //        outputfile << 0 << endl;
                        //        FVClauseNum++;
                        //    }

                        //    else if(RemainFV[i] != 0){
                        for(auto m = 0; m < int(combAtLeastXi.size()); m++){
                        	expr tmp = c.bool_const("tmp");
                        	tmp = c.bool_val(false);
                            for(auto l = 0; l < int(ZVar.size()); l++){
                                outputfile << ZVar[l] << " ";

                                if(ZVar[l] > 0){
                                	int index = ZVar[l] - ZVariable[0];
                                	tmp = tmp || (Z_Variable_Z3[index]);
                                }
                                else if(ZVar[l] < 0){
                                	int index = -ZVar[l] - ZVariable[0];
                                	tmp = tmp || (!Z_Variable_Z3[index]);
                                }
                            }

                            for(auto n = 0; n < int(combAtLeastXi[m].size()); n++){
                                int l = XVar[combAtLeastXi[m][n]-1];
                                int index = XVar[combAtLeastXi[m][n]-1] - X_unknown[gate - 1][0];
                                outputfile << l << " ";

                                tmp = tmp || (X_Unknown_Z3[gate-1][index]);
                            }
                            outputfile << 0 << endl;

                            s.add(tmp);
                            FVClauseNum++;
                        }

                        for (unsigned int p = 0; p < combAtMostXi.size(); p++) {
                        	expr tmp = c.bool_const("tmp");
                        	tmp = c.bool_val(false);
                            for(auto l = 0; l < int(ZVar.size()); l++){
                                outputfile << ZVar[l] << " ";

                                if(ZVar[l] > 0){
                                	int index = ZVar[l] - ZVariable[0];
                                	tmp = tmp || (Z_Variable_Z3[index]);
                                }
                                else if(ZVar[l] < 0){
                                	int index = -ZVar[l] - ZVariable[0];
                                	tmp = tmp || (!Z_Variable_Z3[index]);
                                }
                            }
                            for (unsigned int q = 0; q < combAtMostXi[p].size(); q++) {
                                int l = XVar[combAtMostXi[p][q]-1];
                                outputfile << -l << " ";

                                int index = XVar[combAtMostXi[p][q]-1] - X_unknown[gate - 1][0];
                                tmp = tmp || (!X_Unknown_Z3[gate-1][index]);
                            }
                            outputfile << 0 << endl;

                            s.add(tmp);
                            FVClauseNum++;
                        }
                    }


                }

                else if(YNum != 0){
//                    cout << "YNum=" << YNum << endl;
//                    cout << "Remain FV=" << RemainFV[i] << endl;

                    // The relationship between Y variables and Z variables
                    for(auto x = 0; x < int(FVIndexofZ[i].size()); x++){
                        int t = FVIndexofZ[i][x] - ZVariable[0];
                        for(auto l = 0; l < (coarseGrain1/coarseGrain2); l++){
                        	int m = AllVarRelation[0][t].first[0] - ZVariable[0];
						    int n = AllVarRelation[0][t].second[l] - YVariable[0];

                            outputfile << -AllVarRelation[0][t].first[0] << " " << AllVarRelation[0][t].second[l] << " " << 0 << endl;

                            s.add(!Z_Variable_Z3[m] || Y_Variable_Z3[n]);
                            FVClauseNum++;
                        }
                    }

                    // The relationship between X variables and Y variables
                    for(auto x = 0; x < int(FVIndexofY[i].size()); x++){
                        int t = FVIndexofY[i][x] - YVariable[0];
                        for(auto l = 0; l < coarseGrain2; l++){
                        	int m = AllVarRelation[1][t].first[0] - YVariable[0];
						    int n = AllVarRelation[1][t].second[l] - X_unknown[gate - 1][0];

                            outputfile << -AllVarRelation[1][t].first[0] << " " << AllVarRelation[1][t].second[l] << " " << 0 << endl;

                            s.add(!Y_Variable_Z3[m] || X_Unknown_Z3[gate-1][n]);
                            FVClauseNum++;
                        }
                    }

                    // Coarse granularity constraint for Z-variables
                    int n2 = ZSumNum, j2 = n2 - (ZNum - 1);
                    vector<vector<int> > combAtLeastZi;
                    // Calculate Cn_(n-(G[i-1)), Mat6 stores different combinations in Cn_(n-(G[i]-1)), where each combination is represented as a row
                    combAtLeastZi = combine(n2, j2);  //which is index in indexOfXn_mInG
//                    cout << "The number of different combinations in combAtLeastZi for G[" << i << "]: " << combAtLeastZi.size() << endl;

                    for (unsigned int p = 0; p < combAtLeastZi.size(); p++) {
                    	expr tmp = c.bool_const("tmp");
                    	tmp = c.bool_val(false);
                        for (unsigned int q = 0; q < combAtLeastZi[p].size(); q++) {
                            int x = combAtLeastZi[p][q]-1;
                            outputfile << FVIndexofZ[i][x] << " ";

                            int index = FVIndexofZ[i][x] - ZVariable[0];
                            tmp = tmp || Z_Variable_Z3[index];
                        }
                        outputfile << 0 << endl;
                        s.add(tmp);
                        FVClauseNum++;
                    }

                    // Ensures that for any gate 𝑖, there are at most G[i] inputs.
                    int n3 = ZSumNum, j3 = ZNum + 1;
                    vector<vector<int> > combAtMostZi;
                    // Calculate Cn_(G[i]+1), Mat7 stores different combinations in Cn_(G[i]+1), where each combination is represented as a row.
                    combAtMostZi = combine(n3, j3);
//                    cout << "The number of different combinations in combAtMostZi for G[" << i << "]: " << combAtMostZi.size() << endl;

                    for (unsigned int p = 0; p < combAtMostZi.size(); p++) {
                    	expr tmp = c.bool_const("tmp");
                    	tmp = c.bool_val(false);
                        for (unsigned int q = 0; q < combAtMostZi[p].size(); q++) {
                            int x = combAtMostZi[p][q]-1;
                            outputfile << -FVIndexofZ[i][x] << " ";

                            int index = FVIndexofZ[i][x] - ZVariable[0];
                            tmp = tmp || (!Z_Variable_Z3[index]);
                        }
                        outputfile << 0 << endl;
                        s.add(tmp);
                        FVClauseNum++;
                    }

                    // Coarse granularity constraint for Y-variables
                    vector< vector<int> > combCoarseSearchZ = ProduceBinComb(ZSumNum, ZNum);
//                    cout << "The number of different combinations for combZi: " << combCoarseSearchZ.size() << endl;

                    // Calculate the number of Y-variables whose values have not been determined
                    int YPartNum = (ZSumNum-ZNum) * (coarseGrain1/coarseGrain2);

                    int n4 = YPartNum, j4 = n4 - (YNum - 1);
                    vector<vector<int> > combAtLeastYi;
                    // Calculate Cn_(n-(G[i-1)), Mat6 stores different combinations in Cn_(n-(G[i]-1)), where each combination is represented as a row
                    combAtLeastYi = combine(n4, j4);  //which is index in indexOfXn_mInG


//                    cout << "The number of different combinations in combAtLeastYi for G[" << i << "]: " << combAtLeastYi.size() << endl;

                    // Ensures that for any gate 𝑖, there are at most G[i] inputs.
                    int n5 = YPartNum, j5 = YNum + 1;
                    vector<vector<int> > combAtMostYi;
                    // Calculate Cn_(G[i]+1), Mat7 stores different combinations in Cn_(G[i]+1), where each combination is represented as a row.
                    combAtMostYi = combine(n5, j5);


//                    cout << "The number of different combinations in combAtMostYi for G[" << i << "]: " << combAtMostYi.size() << endl;

                    for(auto j = 0; j < int(combCoarseSearchZ.size()); j++){
                        vector <int> ZVar{};
                        vector <int> YVar{};
                        for(auto k = 0; k < int(combCoarseSearchZ[j].size()); k++){
                            if(combCoarseSearchZ[j][k] == 0){
                                ZVar.push_back(FVIndexofZ[i][k]);
                                int l = FVIndexofZ[i][k]-ZVariable[0];
                                for(int y = 0; y < (coarseGrain1/coarseGrain2); y++){
                                    YVar.push_back(AllVarRelation[0][l].second[y]);
                                }
                            }
                            else if(combCoarseSearchZ[j][k] == 1){
                                ZVar.push_back(-FVIndexofZ[i][k]);
                            }

                        }

                        for(auto m = 0; m < int(combAtLeastYi.size()); m++){
                        	expr tmp = c.bool_const("tmp");
                        	tmp = c.bool_val(false);
                            for(auto l = 0; l < int(ZVar.size()); l++){
                                outputfile << ZVar[l] << " ";

                                if(ZVar[l] > 0){
                                	int index = ZVar[l] - ZVariable[0];
                                	tmp = tmp || (Z_Variable_Z3[index]);
                                }
                                else if(ZVar[l] < 0){
                                	int index = -ZVar[l] - ZVariable[0];
                                	tmp = tmp || (!Z_Variable_Z3[index]);
                                }
                            }

                            for(auto n = 0; n < int(combAtLeastYi[m].size()); n++){
                                int l = YVar[combAtLeastYi[m][n]-1];
                                int index = YVar[combAtLeastYi[m][n]-1] - YVariable[0];
                                outputfile << l << " ";

                                tmp = tmp || (Y_Variable_Z3[index]);
                            }
                            outputfile << 0 << endl;

                            s.add(tmp);
                            FVClauseNum++;
                        }

                        for (unsigned int p = 0; p < combAtMostYi.size(); p++) {
                        	expr tmp = c.bool_const("tmp");
                        	tmp = c.bool_val(false);
                            for(auto l = 0; l < int(ZVar.size()); l++){
                                outputfile << ZVar[l] << " ";

                                if(ZVar[l] > 0){
                                	int index = ZVar[l] - ZVariable[0];
                                	tmp = tmp || (Z_Variable_Z3[index]);
                                }
                                else if(ZVar[l] < 0){
                                	int index = -ZVar[l] - ZVariable[0];
                                	tmp = tmp || (!Z_Variable_Z3[index]);
                                }
                            }
                            for (unsigned int q = 0; q < combAtMostYi[p].size(); q++) {
                                int l = YVar[combAtMostYi[p][q]-1];
                                outputfile << -l << " ";

                                int index = YVar[combAtMostYi[p][q]-1] - YVariable[0];
                                tmp = tmp || (!Y_Variable_Z3[index]);
                            }
                            outputfile << 0 << endl;

                            s.add(tmp);
                            FVClauseNum++;
                        }
                    }

                    // Fine granularity constraint for X-variables
                    vector< vector<int> > combCoarseSearchY{};

                    // Calculate the one-hot combination of Y-variables
                    vector< vector<int> > combCoarsePartSearchY = ProduceBinComb(YPartNum, YNum);

                    for(auto j = 0; j < int(combCoarseSearchZ.size()); j++){
                        vector <int> YVar(YSumNum, -1);

                        for(auto k = 0; k < int(combCoarseSearchZ[j].size()); k++){
                            if(combCoarseSearchZ[j][k] == 1){
                                int l = FVIndexofZ[i][k]-ZVariable[0];
                                for(int y = 0; y < (coarseGrain1/coarseGrain2); y++){
                                    int Yvalue = AllVarRelation[0][l].second[y];
                                    int YIndex = int(find(FVIndexofY[i].begin(), FVIndexofY[i].end(), Yvalue)- FVIndexofY[i].begin());
                                    YVar[YIndex] = 1;
                                }
                            }
                        }

                        for(auto x = 0; x < int(combCoarsePartSearchY.size()); x++){
                            vector <int> YVarChange(YVar);
                            int index = 0;
                            for(auto y = 0; y < int(YVarChange.size()); y++){
                                if(YVarChange[y] == -1){
                                    YVarChange[y] = combCoarsePartSearchY[x][index];
                                    index++;
                                }
                            }
                            combCoarseSearchY.push_back(YVarChange);
                        }

                    }

                    //                    int YOneNum = (toBeAssignedCV1[i]+toBeAssignedCV2[i])/coarseGrain2;
                    //                    // Calculate the one-hot combination of Y-variables
                    //                    vector< vector<int> > combCoarseSearchY = ProduceBinComb(YPartNum, YNum);

//                    cout << "The number of different combinations for combYi: " << combCoarseSearchY.size() << endl;

                    // Calculate the number of X-variables whose values have not been determined
                    int XSumNum = int(IndexMatrix.second[i].size()) - ZNum*coarseGrain2 - YNum*coarseGrain1;

                    int n6 = XSumNum, j6 = n6 - (RemainFV[i] - 1);
                    vector<vector<int> > combAtLeastXi;
                    // Calculate Cn_(n-(G[i-1)), Mat6 stores different combinations in Cn_(n-(G[i]-1)), where each combination is represented as a row
                    combAtLeastXi = combine(n6, j6);  //which is index in indexOfXn_mInG


//                    cout << "The number of different combinations in combAtLeastXi for G[" << i << "]: " << combAtLeastXi.size() << endl;

                    // Ensures that for any gate 𝑖, there are at most G[i] inputs.
                    int n7 = XSumNum, j7 = RemainFV[i] + 1;
                    vector<vector<int> > combAtMostXi;
                    // Calculate Cn_(G[i]+1), Mat7 stores different combinations in Cn_(G[i]+1), where each combination is represented as a row.
                    combAtMostXi = combine(n7, j7);


//                    cout << "The number of different combinations in combAtMostXi for G[" << i << "]: " << combAtMostXi.size() << endl;

                    for(auto j = 0; j < int(combCoarseSearchY.size()); j++){
                        vector <int> YVar{};
                        vector <int> XVar{};
                        for(auto k = 0; k < int(combCoarseSearchY[j].size()); k++){
                            if(combCoarseSearchY[j][k] == 0){
                                YVar.push_back(FVIndexofY[i][k]);
                                int l = FVIndexofY[i][k]-YVariable[0];
                                for(int x = 0; x < coarseGrain2; x++){
                                    XVar.push_back(AllVarRelation[1][l].second[x]);
                                }
                            }
                            else if(combCoarseSearchY[j][k] == 1){
                                YVar.push_back(-FVIndexofY[i][k]);
                            }

                        }


                        for(auto m = 0; m < int(combAtLeastXi.size()); m++){
                        	expr tmp = c.bool_const("tmp");
                        	tmp = c.bool_val(false);
                            for(auto l = 0; l < int(YVar.size()); l++){
                                outputfile << YVar[l] << " ";

                                if(YVar[l] > 0){
                                	int index = YVar[l] - YVariable[0];
                                	tmp = tmp || (Y_Variable_Z3[index]);
                                }
                                else if(YVar[l] < 0){
                                	int index = -YVar[l] - YVariable[0];
                                	tmp = tmp || (!Y_Variable_Z3[index]);
                                }
                            }

                            for(auto n = 0; n < int(combAtLeastXi[m].size()); n++){
                                int l = XVar[combAtLeastXi[m][n]-1];
                                int index = XVar[combAtLeastXi[m][n]-1] - X_unknown[gate - 1][0];

                                outputfile << l << " ";
                                tmp = tmp || (X_Unknown_Z3[gate-1][index]);
                            }
                            outputfile << 0 << endl;

                            s.add(tmp);
                            FVClauseNum++;
                        }

                        for (unsigned int p = 0; p < combAtMostXi.size(); p++) {
                        	expr tmp = c.bool_const("tmp");
                        	tmp = c.bool_val(false);
                            for(auto l = 0; l < int(YVar.size()); l++){
                                outputfile << YVar[l] << " ";

                                if(YVar[l] > 0){
                                	int index = YVar[l] - YVariable[0];
                                	tmp = tmp || (Y_Variable_Z3[index]);
                                }
                                else if(YVar[l] < 0){
                                	int index = -YVar[l] - YVariable[0];
                                	tmp = tmp || (!Y_Variable_Z3[index]);
                                }
                            }
                            for (unsigned int q = 0; q < combAtMostXi[p].size(); q++) {
                                int l = XVar[combAtMostXi[p][q]-1];
                                outputfile << -l << " ";

                                int index = XVar[combAtMostXi[p][q]-1] - X_unknown[gate - 1][0];
                                tmp = tmp || (!X_Unknown_Z3[gate-1][index]);
                            }
                            outputfile << 0 << endl;

                            s.add(tmp);
                            FVClauseNum++;
                        }
                    }
                }

            }
        }

    }
//    cout << "The number of feature vector clauses:" << FVClauseNum << endl;
}

//void ConstructClause::ConstructFVClause(string prefix, ofstream& outputfile, vector <int> featureVector, vector <int> largestCubeVector, vector<int> toBeAssignedCV, vector <IndexRelation> AllVarRelation, int coarseGrain, MatInfo IndexMatrix){
//    int FVClauseNum = 0;
//
//    // Determine the index in G for each Y-variable
//    vector < vector <int> > FVIndexofY (variableNumber+1, vector <int> {});
//    for(auto p = 0; p < int(FVIndexofY.size()); p++){
//        for(auto q = 0; q < int(AllVarRelation.size()); q++){
//            if(AllVarRelation[q].first[1] == p){
//                FVIndexofY[p].push_back(AllVarRelation[q].first[0]);
//            }
//        }
//    }
//
//    // Determine the internal RFV, update Remaining FV after inserting the cubes with largest CV in the truth table
//    vector <int> InternalRFV(variableNumber+1);
//
//    for(auto i = 0; i < int(featureVector.size()); i++){
//        InternalRFV[i] = featureVector[i] - largestCubeVector[i];
//
//        // If featureVector[i] is equal to the maxFV[i], all values equal to 1.
//        if(featureVector[i] == int((IndexMatrix.second)[i].size())){
//            InternalRFV[i] = 0;
//            //            for(auto m = 0; m < FVIndexofY[i].size(); m++){
//            //                int index = FVIndexofY[i][m] - YVariable[0];
//            //                outputfile << YVariable[index] << " " << 0 << endl;
//            //                FVClauseNum++;
//            //            }
//            for(auto j = 0; j < int((IndexMatrix.first)[i].size()); j++){
//                for(auto k = 0; k < int(AssMat.size()); k++){
//                    AssMat[k][IndexMatrix.first[i][j]] = 1;
//                }
//            }
//        }
//
//        // If Internal RemainFV[i] is 0, set elements (i.e., value hasn't been determined) in the corresponding columns(i.e., G[i]) to 0.
//        else if(InternalRFV[i] == 0 && featureVector[i] != int((IndexMatrix.second)[i].size())){
//            for(auto j = 0; j < int((IndexMatrix.first)[i].size());j++){  // Tranverse each column
//                for(auto k = 0; k < int(AssMat.size()); k++){
//                    if(AssMat[k][IndexMatrix.first[i][j]] == -1){
//                        AssMat[k][IndexMatrix.first[i][j]] = 0;
//                    }
//                }
//            }
//        }
//
//    }
//
//    cout << "Assignment matrix after inserting the cube with largest cube vector:" << endl;
//    for(auto i = 0; i < int(AssMat.size()); i++){
//        for(auto j = 0; j < int(AssMat[i].size()); j++){
//            cout << AssMat[i][j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl << endl;
//
//    for(auto p = 0; p < int(AssMat[0].size()); p++){
//        for(auto q = 0; q < int(AssMat.size()); q=q+coarseGrain){
//            if(AssMat[q][p] == 0){
//                int l = (p * int(pow(2, accuracy)) + q)/coarseGrain;
//                outputfile << -YVariable[l] << " " << 0 << endl;
//                FVClauseNum++;
//            }
//
//            if(AssMat[q][p] == 1){
//                int l = (p * int(pow(2, accuracy)) + q)/coarseGrain;
//                outputfile << YVariable[l] << " " << 0 << endl;
//                FVClauseNum++;
//            }
//        }
//    }
//
//    // Determine the final remaining feature vector.(fine-grained search)
//    vector <int> RemainFV(variableNumber+1);
//    for(auto k = 0; k < int(RemainFV.size()); k++){
//        RemainFV[k] = featureVector[k] - largestCubeVector[k] - toBeAssignedCV[k];
//    }
//
//    // For the element 0/1 in the Assignment matrix, construct CNF clauses for corresponding X-variables
//    for(auto p = 0; p < int(AssMat[0].size()); p++){
//        for(auto q = 0; q < int(AssMat.size()); q++){
//            if(AssMat[q][p] == 0){
//                int index = p * int(pow(2, accuracy)) + q;
//                outputfile << -X_unknown[gate-1][index] << " " << 0 << endl;
//                FVClauseNum++;
//            }
//
//            else if(AssMat[q][p] == 1){
//                int index = p * int(pow(2, accuracy)) + q;
//                outputfile << X_unknown[gate-1][index] << " " << 0 << endl;
//                FVClauseNum++;
//            }
//
//        }
//    }
//
//    // Formulate CNF clauses to satisfy feature vector
//    for(auto i = 0; i < int(featureVector.size()); i++){
//        if(InternalRFV[i]!=0){
//
//            // The total number of Y-vars in G[i]
//            int YSumNum = int(FVIndexofY[i].size());
//            // The number of Y-vars for which need to assign 1.
//            int YNum = toBeAssignedCV[i]/coarseGrain;
//            // Calculate the one-hot combination of Y-variables
//            vector< vector<int> > combCoarseSearch = ProduceBinComb(YSumNum, YNum);
//
//            // The total number of X-variables
//            int XVarNum = YSumNum * coarseGrain;
//            // Calculate the number of X-variables whose values have not been determined
//            int XSumNum = (YSumNum-YNum)*coarseGrain;
//            // Calculate the one-hot combination of X-variables whose values have not been determined
//            vector< vector<int> > combFineSearch = ProduceBinComb(XSumNum, RemainFV[i]);
//
//            // Store SOP matrix of Y-var and X-var
//            vector< vector<int> > SOPMat{};
//
//            //Store SOP of Y-var
//            vector <int> YSOP{};
//
//            // Store SOP of all X-variables
//            vector <int> XSOP(XVarNum,-1);
//
//            if(!combCoarseSearch.empty()){
//                // For each one-hot combination of Y-variables
//                for(auto j = 0; j < int(combCoarseSearch.size()); j++){
//                    YSOP.clear();
//                    // Store indexes of all X-variables
//                    vector <int> sizeN(XVarNum);
//                    for (auto p = 0; p < XVarNum; p++) {
//                        sizeN[p] = p;
//                    }
//
//                    int XNum = 0;  // The number of X-vars whose values equal to 1.
//                    for(auto k = 0; k < int(combCoarseSearch[j].size()); k++){
//                        YSOP.push_back(combCoarseSearch[j][k]);
//
//                        // Determine the relationship between X-vars and Y-vars. If Y0=1, then x0 = 1, ……,x(coarseGrain) = 1.
//                        if(combCoarseSearch[j][k] == 1){
//                            //                        int YIndex = FVIndexofY[i][k]-FVIndexofY[0][0];
//                            //                        for(auto l = 0; l < int(AllVarRelation[YIndex].second.size());l++){
//                            //                            int index = AllVarRelation[YIndex].second[l] - X_unknown[gate-1][0];
//                            //                            XSOP[index] = 1;
//                            //
//                            //                            // Delete the index of X-varlables whose value have been determined
//                            //                            sizeN.erase(sizeN.begin()+index-l);
//                            //                        }
//
//                            for(auto l = 0; l < coarseGrain;l++){
//                                // The index of X-vars whose values equal to 1.
//                                int index = k * coarseGrain + l;
//                                XSOP[index] = 1;
//
//                                // Delete the index of X-varlables whose value have been determined
//                                sizeN.erase(sizeN.begin() + index - XNum);
//                                XNum++;
//                            }
//
//                        }
//
//                    }
//
//                    if(!combFineSearch.empty()){
//                        for(auto m = 0; m < int(combFineSearch.size()); m++){
//                            int index = 0;
//                            for(auto n = 0; n < int(combFineSearch[m].size()); n++){
//                                // Determine the values of X-varlables whose value have been determined
//                                XSOP[sizeN[index]] = combFineSearch[m][n];
//                                index++;
//                            }
//
//                            // Store SOP of all variables
//                            vector <int> SOPVec;
//                            // merge(YSOP.begin(), YSOP.end(), XSOP.begin(), XSOP.end(),back_inserter(SOPVec));
//
//                            // Merge SOP of Y-var and SOP of X-var
//                            for(auto l = 0; l < int(YSOP.size())+ int(XSOP.size()); l++){
//                                if(l < int(YSOP.size())){
//                                    SOPVec.push_back(YSOP[l]);
//                                }
//                                else{
//                                    SOPVec.push_back(XSOP[l-int(YSOP.size())]);
//                                }
//                            }
//                            SOPMat.push_back(SOPVec);
//                            SOPVec.clear();
//                        }
//                    }
//                    else{
//
//                        for(auto x = 0; x < int(sizeN.size()); x++){
//                            XSOP[sizeN[x]] = 0;
//                        }
//
//                        // Store SOP of all variables
//                        vector <int> SOPVec;
//
//                        // Merge SOP of Y-var and SOP of X-var
//                        for(auto l = 0; l < int(YSOP.size())+ int(XSOP.size()); l++){
//                            if(l < int(YSOP.size())){
//                                SOPVec.push_back(YSOP[l]);
//                            }
//                            else{
//                                SOPVec.push_back(XSOP[l-int(YSOP.size())]);
//                            }
//                        }
//                        SOPMat.push_back(SOPVec);
//                        SOPVec.clear();
//                    }
//
//                }
//            }
//
//            else{
//                //                vector <int> YSOP(YSumNum, 0);
//                for(auto m = 0; m < FVIndexofY[i].size(); m++){
//                    int index = FVIndexofY[i][m] - YVariable[0];
//                    outputfile << -YVariable[index] << " " << 0 << endl;
//                    FVClauseNum++;
//                }
//
//                for(auto m = 0; m < int(combFineSearch.size()); m++){
//                    int index = 0;
//                    for(auto n = 0; n < int(combFineSearch[m].size()); n++){
//                        // Determine the values of X-varlables whose value have been determined
//                        XSOP[index] = combFineSearch[m][n];
//                        index++;
//                    }
//                    SOPMat.push_back(XSOP);
//
//                    //                    // Store SOP of all variables
//                    //                    vector <int> SOPVec;
//                    //
//                    //                    // Merge SOP of Y-var and SOP of X-var
//                    //                    for(auto l = 0; l < int(YSOP.size())+ int(XSOP.size()); l++){
//                    //                        if(l < int(YSOP.size())){
//                    //                            SOPVec.push_back(YSOP[l]);
//                    //                        }
//                    //                        else{
//                    //                            SOPVec.push_back(XSOP[l-int(YSOP.size())]);
//                    //                        }
//                    //                    }
//                    //                    SOPMat.push_back(SOPVec);
//                    //                    SOPVec.clear();
//                }
//            }
//
//            // Output the pla format of SOP matrix
//            cout << "The PLA format of SOP matrix:" << endl;
//            for (auto p = 0; p < int(SOPMat.size()); p++) {
//                for (auto q = 0; q < int(SOPMat[p].size()); q++) {
//                    cout << SOPMat[p][q] << " ";
//                }
//                cout << endl;
//            }
//            cout << endl << endl;
//
//            // create a temporary .pla file
//            ofstream ofs(prefix+"sop.pla");
//            ofs << ".i " << int(SOPMat[0].size()) << endl;
//            ofs << ".o 1" << endl;
//            for (auto line = 0; line < int(SOPMat.size()); ++line)
//            {
//                for (auto col = 0; col < int(SOPMat[line].size()); ++col)
//                {
//                    ofs << SOPMat[line][col];
//                }
//                ofs << " " << 1 << endl;
//            }
//            ofs << ".e" << endl;
//
//
//            system("/Users/apple/Desktop/Test/SATDifferentGranularities/SATDifferentGranularities/espresso -epos /Users/apple/Desktop/Test/SATDifferentGranularities/SATDifferentGranularities/sop.pla > /Users/apple/Desktop/Test/SATDifferentGranularities/SATDifferentGranularities/pos.pla");
//
//            //          system("./espresso -epos sop.pla > pos.pla");
//
//            auto ifs = ifstream(prefix+"pos.pla");
//            auto line = string();
//            auto count = 0;
//            auto POSRow = 0;
//            auto str = string();
//
//            cout << "The feature vector clause for G[" << i << "]:" << endl;
//            while(getline(ifs, line)){
//                if(count <= 3){
//                    //                    cout << line << endl;
//                    if(count == 3){
//                        stringstream ss;
//                        ss << line;
//                        ss >> str >> POSRow;
//                        //POSRow = int(line[line.size()-1]) - 48;
//                        cout << "The number of POS: " << POSRow << endl;
//                        cout << endl << endl;
//                    }
//                }
//                else if(count == 3 + POSRow +1){
//                    break;
//                }
//                else{
//                    for(auto j = 0; j < int(YSOP.size()) + int(XSOP.size()); j++){
//                        int index = int(line[j])-48;
//
//                        if(j < int(YSOP.size())){
//                            int YIndex = FVIndexofY[i][j]-YVariable[0];
//
//                            if(index == 0){
//                                outputfile << YVariable[YIndex] << " ";
//                            }
//                            else if(index == 1){
//                                outputfile << -YVariable[YIndex] << " ";
//
//                            }
//                            else{
//                                continue;
//                            }
//                        }
//
//                        else{
//                            int XIndex = IndexMatrix.second[i][j-int(YSOP.size())];
//
//                            if(index == 0){
//                                outputfile << X_unknown[gate - 1][XIndex] << " ";
//                            }
//                            else if(index == 1){
//                                outputfile << -X_unknown[gate - 1][XIndex] << " ";
//
//                            }
//                            else{
//                                continue;
//                            }
//                        }
//
//                    }
//                    outputfile << 0 << endl;
//                    FVClauseNum++;
//                }
//                count ++;
//
//            }
//
//        }
//
//    }
//    cout << "The number of feature vector clauses:" << FVClauseNum << endl;
//
//}

// Construct 2nd contraint clause: satisfy feature vector
void ConstructClause::Construct2ndClause(ofstream& outputfile, vector < vector <int> > indexOfXn_mInG, vector <int> largestCubeVector, vector <int> cubeVec){
    int secondClauseNum = 0;

    for (unsigned int i = 0; i < featureVector.size(); i++) {
        // If the value in the FV is equal to the largest value in the Number array
        assert((featureVector[i] >= 0) && (featureVector[i] <= int(indexOfXn_mInG[i].size())));

        if (featureVector[i] == indexOfXn_mInG[i].size()) {
            cout << "G[" << i << "]" << " " << "reaches the maximum." << endl;
            for (unsigned int j = 0; j < indexOfXn_mInG[i].size(); j++) {
                //                cout << X_unknown[gate - 1][indexOfXn_mInG[i][j]] << " " << 0 << endl;
                outputfile << X_unknown[gate - 1][indexOfXn_mInG[i][j]] << " " << 0 << endl;
                secondClauseNum++;
            }
        }

        // If the value in the FV is 0
        else if (featureVector[i] == 0) {
            cout << "G[" << i << "]" << " " << "is 0." << endl;
            for (unsigned int j = 0; j < indexOfXn_mInG[i].size(); j++) {
                //                cout << -X_unknown[gate - 1][indexOfXn_mInG[i][j]] << " " << 0 << endl;
                outputfile << -X_unknown[gate - 1][indexOfXn_mInG[i][j]] << " " << 0 << endl;
                secondClauseNum++;
            }
        }

        else{
            int remainFV = featureVector[i] - largestCubeVector[i];

            if(largestCubeVector[i] != 0){

                // Find the element of indexOfXn_mInG[i] whose value is 1, output them into the CNF file
                vector <int> indexOne{};
                for(auto p = 0; p < int(indexOfXn_mInG[i].size());p++){
                    if(cubeVec[indexOfXn_mInG[i][p]] == 1){
                        indexOne.push_back(p);
                        //                        cout << X_unknown[gate - 1][indexOfXn_mInG[i][p]] << " " << 0 << endl;
                        outputfile << X_unknown[gate - 1][indexOfXn_mInG[i][p]] << " " << 0 << endl;
                        secondClauseNum++;
                    }
                }


                // Delete the elements in indexOfXn_mInG[i] whose value is 1
                for(auto p = 0; p < int(indexOne.size()); p++){
                    indexOfXn_mInG[i].erase(indexOfXn_mInG[i].begin()+ indexOne[p]- p);
                }
            }

            if(remainFV == 0){
                cout << "The remaining feature vector for G[" << i << "]" << " " << "is 0." << endl;
                for (unsigned int j = 0; j < indexOfXn_mInG[i].size(); j++) {
                    outputfile << -X_unknown[gate - 1][indexOfXn_mInG[i][j]] << " " << 0 << endl;
                    //                    cout << -X_unknown[gate - 1][indexOfXn_mInG[i][j]] << " " << 0 << endl;
                    secondClauseNum++;
                }
            }

            else if(remainFV == 1){
                cout << "The remaining feature vector for G[" << i << "]" << " " << "is 1." << endl;
                // Ensures that for any gate 𝑖, there are at least 1 inputs.
                for (unsigned int j = 0; j < indexOfXn_mInG[i].size(); j++) {
                    outputfile << X_unknown[gate - 1][indexOfXn_mInG[i][j]] << " ";
                    //                    cout << X_unknown[gate - 1][indexOfXn_mInG[i][j]] << " ";
                }
                outputfile << 0 << endl;
                secondClauseNum++;
                //                cout << 0 << endl;

                // Ensures that for any gate 𝑖, there are at most 1 inputs.
                int n3 = int(indexOfXn_mInG[i].size()), j3 = 2;
                vector<vector<int> > combAtMost1;
                // Calculate Cn_2
                combAtMost1 = combine(n3, j3);

                cout << "The number of different combinations for Cn_2 in combAtMost1: " << combAtMost1.size() << endl;

                for (unsigned int p = 0; p < combAtMost1.size(); p++) {
                    for (unsigned int q = 0; q < combAtMost1[p].size(); q++) {
                        int x = combAtMost1[p][q]-1;
                        int l = indexOfXn_mInG[i][x];
                        outputfile << -X_unknown[gate - 1][l] << " ";
                        //                        cout << -X_unknown[gate - 1][l] << " ";
                    }
                    outputfile << 0 << endl;
                    secondClauseNum++;
                    //                    cout << 0 << endl;
                }

            }

            else{
                cout << "The remaining feature vector for G[" << i << "] is" << " " << remainFV << endl;
                // Ensures that for any gate 𝑖, there are at least G[i] inputs.
                int n4 = int(indexOfXn_mInG[i].size()), j4 = n4 - (remainFV - 1);
                vector<vector<int> > combAtLeastGi;
                // Calculate Cn_(n-(G[i-1)), Mat6 stores different combinations in Cn_(n-(G[i]-1)), where each combination is represented as a row
                combAtLeastGi = combine(n4, j4);  //which is index in indexOfXn_mInG


                cout << "The number of different combinations for Cn_(n-(G[i]-1)) in combAtLeastGi for G[" << i << "]: " << combAtLeastGi.size() << endl;

                for (unsigned int p = 0; p < combAtLeastGi.size(); p++) {
                    for (unsigned int q = 0; q < combAtLeastGi[p].size(); q++) {
                        int x = combAtLeastGi[p][q]-1;
                        int l = indexOfXn_mInG[i][x];
                        outputfile << X_unknown[gate - 1][l] << " ";
                        //                        cout << X_unknown[gate - 1][l] << " ";
                    }
                    outputfile << 0 << endl;
                    secondClauseNum++;
                    //                    cout << 0 << endl;
                }

                // Ensures that for any gate 𝑖, there are at most G[i] inputs.
                int n5 = int(indexOfXn_mInG[i].size()), j5 = remainFV + 1;
                vector<vector<int> > combAtMostGi;
                // Calculate Cn_(G[i]+1), Mat7 stores different combinations in Cn_(G[i]+1), where each combination is represented as a row.
                combAtMostGi = combine(n5, j5);


                cout << "The number of different combinations for Cn_(G[i]+1) in combAtMostGi for G[" << i << "]: " << combAtMostGi.size() << endl;


                for (unsigned int p = 0; p < combAtMostGi.size(); p++) {
                    for (unsigned int q = 0; q < combAtMostGi[p].size(); q++) {
                        int x = combAtMostGi[p][q]-1;
                        int l = indexOfXn_mInG[i][x];
                        outputfile << -X_unknown[gate - 1][l] << " ";
                        //                        cout << -X_unknown[gate - 1][l] << " ";
                    }
                    outputfile << 0 << endl;
                    secondClauseNum++;
                    //                    cout << 0 << endl;
                }

            }

        }

    }
    cout << "The number of second clauses:" << secondClauseNum << endl;
}

// Determine the gate library
void ConstructClause::GateLibrary(ofstream& outputfile, context& c, solver& s) {
	int gateLibraryClauseNum = 0;
    for (auto i = 0; i < gate; i++) {
//        // Add 12 2-input gates
//        outputfile << f[i][0] << " " << -f[i][1] << " " << f[i][2] << " " << f[i][3] << " " << 0 << endl;
//        outputfile << f[i][0] << " " << f[i][1] << " " << -f[i][2] << " " << f[i][3] << " " << 0 << endl;
//        outputfile << -f[i][0] << " " << -f[i][1] << " " << f[i][2] << " " << -f[i][3] << " " << 0 << endl;
//        outputfile << -f[i][0] << " " << f[i][1] << " " << -f[i][2] << " " << -f[i][3] << " " << 0 << endl;
//
//        s.add(f_Z3[i][0] || !f_Z3[i][1] || f_Z3[i][2] || f_Z3[i][3]);
//        s.add(f_Z3[i][0] || f_Z3[i][1] || !f_Z3[i][2] || f_Z3[i][3]);
//        s.add(!f_Z3[i][0] || !f_Z3[i][1] || f_Z3[i][2] || !f_Z3[i][3]);
//        s.add(!f_Z3[i][0] || f_Z3[i][1] || !f_Z3[i][2] || !f_Z3[i][3]);
    	    auto ifs = ifstream("pos.pla");
			auto line = string();
			auto count = 0;
			auto POSRow = 0;
			auto str = string();

			while(getline(ifs, line)){
				if(count <= 3){
//					cout << line << endl;
					if(count == 3){
						stringstream ss;
						ss << line;
						ss >> str >> POSRow;
						//POSRow = int(line[line.size()-1]) - 48;
	//					cout << "The number of POS: " << POSRow << endl;
	//					cout << endl << endl;
					}
				}

				else if(count == 3 + POSRow +1){
					break;
				}
//				else if(count > 30){
//					break;
//				}
				else{
					expr tmp = c.bool_const("tmp");
					tmp = c.bool_val(false);
					for(auto j = 0; j < int(pow(2, N)); j++){
						int index = int(line[j])-48;

						if(index == 0){
							outputfile << f[i][j] << " ";
							tmp = tmp || f_Z3[i][j];
						}
						else if(index == 1){
							outputfile << -f[i][j] << " ";
							tmp = tmp || (!f_Z3[i][j]);
						}
						else{
							continue;
						}
					}
					outputfile << 0 << endl;
					s.add(tmp);
					gateLibraryClauseNum++;
				}
				count++;
			}
    }
//    cout << "The number of gate library clauses:" << gateLibraryClauseNum << endl;
}

void ConstructClause::GateIndexConstraintZ3(ofstream& outputfile, int gateKind, context& c, solver& s){
	int gateIndexClauseNum = 0;
	for(auto i = 0; i < gate; i++){
		//Add gate index constraints

		//Each gate has exactly one kind of gate.
		//At least 1
		expr tmp = c.bool_const("tmp");
		tmp = c.bool_val(false);
		for(auto k = 0; k < gateKind; k++){
			tmp = tmp || GateIndex_Z3[i][k];
		}
		s.add(tmp);
		gateIndexClauseNum++;

		//At most 1
		int n1 = gateKind, j1 = 2;
		vector< vector<int> > combAtMost1;
		// Calculate Cn_(G[i]+1), Mat7 stores different combinations in Cn_(G[i]+1), where each combination is represented as a row.
		combAtMost1 = combine(n1, j1);

		// cout << "The number of different combinations in combAtMostXi for G[" << i << "]: " << combAtMostGi.size() << endl;

		//At most 1
		for (unsigned int p = 0; p < combAtMost1.size(); p++) {
			expr tmp = c.bool_const("tmp");
			tmp = c.bool_val(false);
			for (unsigned int q = 0; q < combAtMost1[p].size(); q++) {
				int x = combAtMost1[p][q]-1;
//				outputfile << -trueGateCate[x] << " ";
				tmp = tmp || (!GateIndex_Z3[i][x]);
			}
//			outputfile << 0 << endl;
			s.add(tmp);
			gateIndexClauseNum++;
		}

		// The relationship between f_ipquv and gateIndex_ik
		for(auto j = 0; j < gateKind; j++){
			// First implication
			vector <expr> temp1;
			for(auto k = 0; k < gateTruthTable[0].size(); k++){
				if(gateTruthTable[j][k] == 0){
					temp1.push_back(f_Z3[i][k]);
				}
				else if(gateTruthTable[j][k] == 1){
					temp1.push_back(!f_Z3[i][k]);
				}
			}

			expr tmp = c.bool_const("tmp");
			tmp = c.bool_val(false);
			for(auto m = 0; m < gateTruthTable[0].size(); m++){
//				outputfile << temp1[m] << " ";
				tmp = tmp || temp1[m];
			}
			tmp = tmp || GateIndex_Z3[i][j];
			s.add(tmp);
//			outputfile << trueGateCate[j] << " " << 0 << endl;
			gateIndexClauseNum++;

			// Second implication
			vector <expr> temp2;
			for(auto k = 0; k < gateTruthTable[0].size(); k++){
				if(gateTruthTable[j][k] == 0){
					temp2.push_back(!f_Z3[i][k]);
				}
				else if(gateTruthTable[j][k] == 1){
					temp2.push_back(f_Z3[i][k]);
				}
			}

			for(auto m = 0; m < gateTruthTable[0].size(); m++){
				s.add(!GateIndex_Z3[i][j] || temp2[m]);
//				outputfile << -trueGateCate[j] << " " << temp2[m] << " " << 0 << endl;
				gateIndexClauseNum++;
			}
		}

		// The relationship between GateIndex_Z3 and GateIndex_Z3_INT
		for(auto j = 0; j < gateKind; j++){
			s.add(!GateIndex_Z3[i][j] || (GateIndex_Z3_INT[i][j] == 1));
			s.add( GateIndex_Z3[i][j] || (GateIndex_Z3_INT[i][j] == 0));
		}

	}

//	// Area constraint
//	expr tmp = c.int_const("tmp");
//	tmp = c.int_val(0);
//	for(auto i = 0; i < gate; i++){
//		for(auto j = 0; j < gateKind; j++){
//			tmp = tmp + gateArea[j] * (GateIndex_Z3_INT[i][j]);
//		}
//	}
//	s.add(tmp <= maxArea);

//	cout << "The number of gate index clauses:" << gateIndexClauseNum << endl;
}

void ConstructClause::areaConstraintZ3(ofstream& outputfile, int gateKind, vector <double> gateArea, double maxArea, context& c, solver& s){
//	int gateAreaClauseNum = 0;
	// Area constraint
	string max_area1 = to_string(maxArea);
	const char* max_area2 = max_area1.c_str();
	expr tmp = c.real_const("tmp");
	tmp = c.real_val(0);
	for(auto i = 0; i < gate; i++){
		for(auto j = 0; j < gateKind; j++){
			string gateArea_str = to_string(gateArea[j]);
			const char* gateArea_char = gateArea_str.c_str();
//			tmp = tmp + gateArea[j] * (GateIndex_Z3_INT[i][j]);
			tmp = tmp + c.real_val(gateArea_char) * (GateIndex_Z3_INT[i][j]);
		}
	}
	s.add(tmp <= c.real_val(max_area2));
//	s.add(tmp <= c.real_val("9.5"));

//	cout << "The number of gate index clauses:" << gateIndexClauseNum << endl;
}

void ConstructClause::GateIndexConstraint(ofstream& outputfile, int gateKind, vector <int> gateArea){
    int gateIndexClauseNum = 0;
    for(auto i = 0; i < gate; i++){
        // Obtain the true gate index from whole gate indexes
        vector <int> trueGateCate(gateKind, 0);
        int index = 0;
        for(auto j = 0; j < gateKind; j++){
            trueGateCate[j] = GateIndex[i][index];
            // cout << trueGateCate[j] << " ";
            index = index + gateArea[j];
        }
        // cout << endl;

        //Add gate index constraints

        //Each gate has exactly one kind of gate.
        //At least 1
        for(auto k = 0; k < trueGateCate.size(); k++){
            outputfile << trueGateCate[k] << " ";
        }
        outputfile << 0 << endl;
        gateIndexClauseNum++;

        //At most 1

        int n1 = gateKind, j1 = 2;
        vector< vector<int> > combAtMostGi;
        // Calculate Cn_(G[i]+1), Mat7 stores different combinations in Cn_(G[i]+1), where each combination is represented as a row.
        combAtMostGi = combine(n1, j1);

        // cout << "The number of different combinations in combAtMostXi for G[" << i << "]: " << combAtMostGi.size() << endl;

        for (unsigned int p = 0; p < combAtMostGi.size(); p++) {
            for (unsigned int q = 0; q < combAtMostGi[p].size(); q++) {
                int x = combAtMostGi[p][q]-1;
                outputfile << -trueGateCate[x] << " ";
            }
            outputfile << 0 << endl;
            gateIndexClauseNum++;
        }

        // Multiple variables are equal for one gate kind
        for(auto j = 0; j < gateKind; j++){
            if(gateArea[j] > 1){
                for(auto k = trueGateCate[j]+1; k < trueGateCate[j+1]; k++){
                    outputfile << -trueGateCate[j] << " " << k << " " << 0 << endl;
                    gateIndexClauseNum++;
                    outputfile << trueGateCate[j] << " " << -k << " " << 0 << endl;
                    gateIndexClauseNum++;
                }
            }
        }

        // The relationship between f_ipq and v_ik
        for(auto j = 0; j < gateKind; j++){
            // First implication
            vector <int> temp1(gateTruthTable[0].size(), 0);
            for(auto k = 0; k < gateTruthTable[0].size(); k++){
                if(gateTruthTable[j][k] == 0){
                    temp1[k] = f[i][k];
                }
                else if(gateTruthTable[j][k] == 1){
                    temp1[k] = -f[i][k];
                }
            }

            for(auto m = 0; m < temp1.size(); m++){
                outputfile << temp1[m] << " ";
            }
            outputfile << trueGateCate[j] << " " << 0 << endl;
            gateIndexClauseNum++;

            // Second implication
            vector <int> temp2(gateTruthTable[0].size(), 0);
            for(auto k = 0; k < gateTruthTable[0].size(); k++){
                if(gateTruthTable[j][k] == 0){
                    temp2[k] = -f[i][k];
                }
                else if(gateTruthTable[j][k] == 1){
                    temp2[k] = f[i][k];
                }
            }

            for(auto m = 0; m < temp2.size(); m++){
                outputfile << -trueGateCate[j] << " " << temp2[m] << " " << 0 << endl;
                gateIndexClauseNum++;
            }
        }

        // Area constraint

    }

    cout << "The number of gate index clauses:" << gateIndexClauseNum << endl;
}

string ConstructClause::SATSolver(int tryNum, int maxIterationPerRun, string prefix, ofstream& outputfile, vector < vector <int> > indexOfXn_mInG){
    int runNum = 0;
    int solutionNum = 0;
    auto result = string();

    auto line = string();
    stringstream ss;

    ofstream solutions;
    solutions.open(prefix+"solutions.cnf");

    // while(runNum == 0 || result == "SATISFIABLE"){
    while(runNum == 0 || result == "SAT"){
        runNum++;
        if(runNum == maxIterationPerRun+1){
            break;
        }

        // char command[1000];
        // strcpy(command, prefix.data());
        // strcat(command, "minisat ");
        // strcat(command, prefix.data());
        // strcat(command, "test53.cnf ");
        // strcat(command, prefix.data());
        // strcat(command, "result0.cnf");
        // system(command);
        //        system("/Users/apple/Desktop/Test/Project1/Project1/minisat /Users/apple/Desktop/Test/Project1/Project1/test53.cnf /Users/apple/Desktop/Test/Project1/Project1/result0.cnf");

        system("minisat test53.cnf result0.cnf");
//        system("/Users/apple/Desktop/Test/ThreeGranuWithLCV/ThreeGranuWithLCV/painless-mcomsps test53.cnf > result0.cnf");
    //    system("painless-mcomsps test53.cnf > result0.cnf");

        ifstream in;
        in.open(prefix+"result0.cnf");

    //    ofstream out;
    //    out.open(prefix+"result.cnf");
    //    auto line = string();

    //    string value;

    //    if(in){
    //        while(getline(in,line)){
    //            if(line[0] == 'c'){
    //                cout << line << endl;
    //            }

    //            if(line[0] == 's'){
    //                stringstream ss(line);
    //                while(ss >> value){
    //                    if(value != "s"){
    //                        result = value;
    //                        cout << result << endl;
    //                    }
    //                }

    //            }
    //            if(line[0] == 'v'){
    //                stringstream ss(line);
    //                while(ss >> value){
    //                    if(value != "v"){
    //                        solutions << value << " ";
    //                        out << value << " ";
    //                    }
    //                }
    //            }
    //        }
    //    }
    //    solutions << endl;
    //    out << endl;


        ss.str(string());
        ss.clear();
        getline(in, line); // first line: SAT or UNSAT
        ss << line;
        ss >> result;
        // cout << result << endl;

        // ifstream input;
        // input.open(prefix+"result.cnf");

        // if(result == "SATISFIABLE"){
        if(result == "SAT"){
            // solutionNum = PrintCircuit(prefix,indexOfXn_mInG);
            solutionNum++;
            while(in >> line) {
                int num = atoi(line.c_str());
                solutions << num << " ";
                num = num * (-1);
                outputfile << num << " ";
            }
            solutions << endl;
            outputfile << endl;
        }
    }
    cout << "The number of solutions for " << tryNum << "th cube is: "<< solutionNum;
    return result;
}

int ConstructClause::SMTSolver(context& c, solver& s, int maxIterationPerRun, vector < vector <int> > indexOfXn_mInG, vector <double> gateArea){
	int runNum = 0;
	auto result = sat;
	while(runNum == 0 || result == sat){
	        runNum++;
	        if(runNum == maxIterationPerRun+1){
	            break;
	        }

	        params p(c);
	        p.set(":timeout", 60000u);
	        s.set(p);

	        result = s.check();
	        if(result == unknown){
	        	result = unsat;
	        }
	        cout << result << endl;
//	        g_condWait.notify_one();
//	        model m = s.get_model();
	        // cout << m << "\n";
//	        PrintVerifyCircuitZ3(m, indexOfXn_mInG, gateArea);
//	        cout << endl << endl;
//
//        	expr tmp = c.bool_const("tmp");
//        	tmp = c.bool_val(false);
//
////			        for(auto i = 0; i < m.size(); i++){
////			        	func_decl v = m[i];
////			        	tmp = tmp || (v != m.get_const_interp(v));
////			        }
//
//	        for(auto i = 0; i < X_Unknown_Z3.size(); i++){
//	        	for(auto j = 0; j < X_Unknown_Z3[i].size(); j++){
//
//					tmp = tmp || (X_Unknown_Z3[i][j] != m.eval(X_Unknown_Z3[i][j]));
//	        	}
//	        }
//	        s.add(tmp);
	}
	if(result == sat){
		return 1;
	}
	else if(result == unsat || result == unknown){
		return 0;
	}
}

double ConstructClause::PrintVerifyCircuitZ3(model& m, vector < vector <int> > indexOfXn_mInG, vector <double> gateArea, int solutionNum, vector <int>& areaSet){
	// 	Verify Circuit
	int index = (gate-1) * int(pow(2, accuracy+variableNumber));
	vector <int> calFeatuteVec(variableNumber+1);
	for(auto j = 0; j < int(indexOfXn_mInG.size()); j++){
		int sum = 0;
		for(auto k = 0; k < int(indexOfXn_mInG[j].size());k++){
			int x = indexOfXn_mInG[j][k];
			auto item = X_Unknown_Z3[gate-1][x];
			if(m.eval(item).is_true()){
				sum++;
			}
		}
		calFeatuteVec[j] = sum;
	}

	int flag = 1;
	for(auto i = 0; i < variableNumber+1; i++){
		if(calFeatuteVec[i] != featureVector[i]){
			flag = 0;
		}
	}

	if(flag == 1){
		cout << "Verify successfully!" << endl;
		cout << endl;
	}
	else{
		cout << "Verify failed! The feature vector is: ";
		for(auto i = 0; i < variableNumber+1; i++){
			cout << calFeatuteVec[i] << " ";
		}
		cout << endl;
	}

	//Print Circuit
	vector < vector <string> > circuitRes (gate, vector <string>(N+2));
	vector <int> circuitIndex(gate, -1);
	for(auto i = 0; i < int(S_Z3.size()); i++){
//		cout << "For gate " << variableNumber + accuracy + i << ":" << endl;
//		cout << "The four fanins are: ";
		circuitRes[i][5] = std::to_string(variableNumber + accuracy + i);

//		auto k = S[i][0];
		auto count = 0;
		for(auto j = 0; j < S_Z3[i].size(); j++){
			auto item = S_Z3[i][j];
			if(m.eval(item).is_true()){
				count++;
//				cout << j << " ";
				if(count == 1){
					circuitRes[i][1] = std::to_string(j);
				}
				else if(count == 2){
					circuitRes[i][2] = std::to_string(j);
				}
				else if(count == 3){
					circuitRes[i][3] = std::to_string(j);
				}
				else if(count == 4){
					circuitRes[i][4] = std::to_string(j);
				}
			}
		}
//		cout << ",the operator is: ";
//		auto p = f[i][0];
		auto gateIndex = string();
		for(auto q = 0; q < int(f_Z3[i].size()); q++){
			auto item = f_Z3[i][q];
			if(m.eval(item).is_true()){
				gateIndex.push_back('1');
				//cout << 1 << " ";
			}
			else{
				gateIndex.push_back('0');
				//cout << 0 << " ";
			}
		}
		//cout << gateIndex << " ";

		// cout << gateIndex << endl;
		auto iter = gateLibrary.find(gateIndex);
//		cout << iter->second << ", ";
//		gateType.push_back(iter->second);
		circuitRes[i][0] = iter->second;

//		cout << "the gate index is: ";

		for(auto k = 0; k < GateIndex_Z3[i].size(); k++){
			auto item  = GateIndex_Z3[i][k];
			if(m.eval(item).is_true()){
//				cout << k << endl;
				circuitIndex[i] = k;
//				circuitRes[i][0] = k;
				//cout << 1 << " ";
			}

//			auto item1  = GateIndex_Z3_INT[i][k];
//			if(m.eval(item1).get_numeral_int()> 0){
//				cout << k << " " << m.eval(item1).get_numeral_int() << endl;
//				//cout << 1 << " ";
//			}
		}

	}

//	cout << endl;

	// create a temporary .v file
	// Output the first three lines for .v file
	char sol[1000];
	 strcpy(sol, "solution");
	 strcat(sol, std::to_string(solutionNum).data());
	 strcat(sol, ".v");
	ofstream ofs(sol);
	ofs << "module \\solution-0 ("<< endl;
	ofs << "  ";
	for(int i = 0; i < variableNumber+accuracy; i++){
		if(i == variableNumber+accuracy-1){
			ofs << "x" << i << "," << endl;
		}
		else{
			ofs << "x" << i << ", ";
		}
	}
	ofs << "  z0 );" << endl;

	ofs << "  input" << " ";
	for(int i = 0; i < variableNumber+accuracy; i++){
		if(i == variableNumber+accuracy-1){
			ofs << "x" << i << ";" << endl;
		}
		else{
			ofs << "x" << i << ", ";
		}
	}
	ofs << "  output z0;" << endl;

	if(gate > 1){
		ofs << "  wire" << " ";
		for(int i = variableNumber+accuracy; i < variableNumber+accuracy+gate-1; i++){
			if(i == variableNumber+accuracy+gate-2){
				ofs << "new_n" << i << "_;" << endl;
			}
			else{
				ofs << "new_n" << i << "_, ";
			}
		}
	}

	for(auto p = 0; p < circuitRes.size(); p++){

		if(circuitIndex[p] == 18 || circuitIndex[p] == 29){
			ofs << "  " << circuitRes[p][0] << "  g" << p;
			if(atoi(circuitRes[p][1].c_str()) < variableNumber+accuracy){
				ofs << "(.a(x" << circuitRes[p][1] << "), ";
			}
			else if(atoi(circuitRes[p][1].c_str()) >= variableNumber+accuracy){
				ofs << "(.a(new_n" << circuitRes[p][1] << "_" << "), ";
			}

			if(atoi(circuitRes[p][2].c_str()) < variableNumber+accuracy){
				ofs << ".b(x" << circuitRes[p][2] << "), ";
			}
			else if(atoi(circuitRes[p][2].c_str()) >= variableNumber+accuracy){
				ofs << ".b(new_n" << circuitRes[p][2] << "_" << "), ";
			}

			if(atoi(circuitRes[p][3].c_str()) < variableNumber+accuracy){
				ofs << ".c(x" << circuitRes[p][3] << "), ";
			}
			else if(atoi(circuitRes[p][3].c_str()) >= variableNumber+accuracy){
				ofs << ".c(new_n" << circuitRes[p][3] << "_" << "), ";
			}

			if(atoi(circuitRes[p][4].c_str()) < variableNumber+accuracy){
				ofs << ".d(x" << circuitRes[p][4] << "), ";
			}
			else if(atoi(circuitRes[p][4].c_str()) >= variableNumber+accuracy){
				ofs << ".d(new_n" << circuitRes[p][4] << "_" << "), ";
			}
		}

		if((circuitIndex[p] >= 66 && circuitIndex[p] <= 68) || (circuitIndex[p] >= 81 && circuitIndex[p] <= 83) ){

			int position1 = circuitRes[p][0].find("-");
			int position2 = circuitRes[p][0].find(",");

			string gateType = circuitRes[p][0].substr(0, position1);

			int a = circuitRes[p][0][position1+1] - '0';
			int b = circuitRes[p][0][position1+2] - '0';

			int c = circuitRes[p][0][position2+1] - '0';
			int d = circuitRes[p][0][position2+2] - '0';

			ofs << "  " << gateType << "  g" << p;
			if(atoi(circuitRes[p][a].c_str()) < variableNumber+accuracy){
				ofs << "(.a(x" << circuitRes[p][a] << "), ";
			}
			else if(atoi(circuitRes[p][a].c_str()) >= variableNumber+accuracy){
				ofs << "(.a(new_n" << circuitRes[p][a] << "_" << "), ";
			}

			if(atoi(circuitRes[p][b].c_str()) < variableNumber+accuracy){
				ofs << ".b(x" << circuitRes[p][b] << "), ";
			}
			else if(atoi(circuitRes[p][b].c_str()) >= variableNumber+accuracy){
				ofs << ".b(new_n" << circuitRes[p][b] << "_" << "), ";
			}

			if(atoi(circuitRes[p][c].c_str()) < variableNumber+accuracy){
				ofs << ".c(x" << circuitRes[p][c] << "), ";
			}
			else if(atoi(circuitRes[p][c].c_str()) >= variableNumber+accuracy){
				ofs << ".c(new_n" << circuitRes[p][c] << "_" << "), ";
			}

			if(atoi(circuitRes[p][d].c_str()) < variableNumber+accuracy){
				ofs << ".d(x" << circuitRes[p][d] << "), ";
			}
			else if(atoi(circuitRes[p][d].c_str()) >= variableNumber+accuracy){
				ofs << ".d(new_n" << circuitRes[p][d] << "_" << "), ";
			}
		}

		else if((circuitIndex[p] >= 14 && circuitIndex[p] <= 17) || (circuitIndex[p] >= 25 && circuitIndex[p] <= 28) || (circuitIndex[p] >= 54 && circuitIndex[p] <= 65) || (circuitIndex[p] >= 69 && circuitIndex[p] <= 80)){

//			int index = -1;
//			string tmp = circuitRes[p][0];
//			for(auto m = 0; m < circuitRes[p][0].length(); m++){
//				if( tmp.at(m) == "-"){
//					index = m;
//				}
//			}

			int position = circuitRes[p][0].find("-");

			string gateType = circuitRes[p][0].substr(0, position);

			int a = circuitRes[p][0][position+1] - '0';
			int b = circuitRes[p][0][position+2] - '0';
			int c = circuitRes[p][0][position+3] - '0';

			ofs << "  " << gateType << "  g" << p;
			if(atoi(circuitRes[p][a].c_str()) < variableNumber+accuracy){
				ofs << "(.a(x" << circuitRes[p][a] << "), ";
			}
			else if(atoi(circuitRes[p][a].c_str()) >= variableNumber+accuracy){
				ofs << "(.a(new_n" << circuitRes[p][a] << "_" << "), ";
			}

			if(atoi(circuitRes[p][b].c_str()) < variableNumber+accuracy){
				ofs << ".b(x" << circuitRes[p][b] << "), ";
			}
			else if(atoi(circuitRes[p][b].c_str()) >= variableNumber+accuracy){
				ofs << ".b(new_n" << circuitRes[p][b] << "_" << "), ";
			}

			if(atoi(circuitRes[p][c].c_str()) < variableNumber+accuracy){
				ofs << ".c(x" << circuitRes[p][c] << "), ";
			}
			else if(atoi(circuitRes[p][c].c_str()) >= variableNumber+accuracy){
				ofs << ".c(new_n" << circuitRes[p][c] << "_" << "), ";
			}
		}

		else if((circuitIndex[p] >= 8 && circuitIndex[p] <= 13) || (circuitIndex[p] >= 19 && circuitIndex[p] <= 24) || (circuitIndex[p] >= 30 && circuitIndex[p] <= 53)){

//			int index = -1;
//			string tmp = circuitRes[p][0];
//			for(auto m = 0; m < circuitRes[p][0].length(); m++){
//				if( tmp.at(m) == "-"){
//					index = m;
//				}
//			}

			int position = circuitRes[p][0].find("-");

			string gateType = circuitRes[p][0].substr(0, position);

			int a = circuitRes[p][0][position+1] - '0';
			int b = circuitRes[p][0][position+2] - '0';

			ofs << "  " << gateType << "  g" << p;
			if(atoi(circuitRes[p][a].c_str()) < variableNumber+accuracy){
				ofs << "(.a(x" << circuitRes[p][a] << "), ";
			}
			else if(atoi(circuitRes[p][a].c_str()) >= variableNumber+accuracy){
				ofs << "(.a(new_n" << circuitRes[p][a] << "_" << "), ";
			}

			if(atoi(circuitRes[p][b].c_str()) < variableNumber+accuracy){
				ofs << ".b(x" << circuitRes[p][b] << "), ";
			}
			else if(atoi(circuitRes[p][b].c_str()) >= variableNumber+accuracy){
				ofs << ".b(new_n" << circuitRes[p][b] << "_" << "), ";
			}
		}

		else if((circuitIndex[p] >= 0 && circuitIndex[p] <= 7)){

			int position = circuitRes[p][0].find("-");

			string gateType = circuitRes[p][0].substr(0, position);

			int a = circuitRes[p][0][position+1] - '0';
//			int b = circuitRes[p][0][position+2] - '0';

			ofs << "  " << gateType << "1" << "  g" << p;
			if(atoi(circuitRes[p][a].c_str()) < variableNumber+accuracy){
				ofs << "(.a(x" << circuitRes[p][a] << "), ";
			}
			else if(atoi(circuitRes[p][a].c_str()) >= variableNumber+accuracy){
				ofs << "(.a(new_n" << circuitRes[p][a] << "_" << "), ";
			}
		}

		if(p < gate-1){
			ofs << ".O(new_n" << variableNumber+accuracy+p << "_));" << endl;
		}

		else if (p == gate-1){
			ofs << ".O(z0));" << endl;
		}
	}
	ofs << "endmodule" << endl;

	cout << "The area of circuit is: ";
	double area = 0;
	for(auto i = 0; i < gate; i++){
		int index = circuitIndex[i];
		area = area + gateArea[index];
	}
	cout << area << endl;
	areaSet[solutionNum] = area;

	 char command[1000];
//	 strcpy(command, prefix.data());
	 strcpy(command, "abc -c \"read mcnc.genlib; read -m solution");
	 strcat(command, std::to_string(solutionNum).data());
	 strcat(command, ".v; write_dot solution");
	 strcat(command, std::to_string(solutionNum).data());
	 strcat(command, ".dot; print_stats\"");
	 system(command);
//	system("abc -c \"read mcnc.genlib; read -m solution.v; write_dot solution.dot; print_stats\"");
	 solutionNum++;
	return area;
}

int ConstructClause::PrintCircuit(string prefix, vector < vector <int> > indexOfXn_mInG){
    ifstream in;
    in.open(prefix+"solutions.cnf");
    auto line = string();
    int count = 0;

    if(in){

        while(getline(in,line)){
            //            cout << "Solution " << count << endl;
            stringstream ss(line);
            vector <int> res;
            int i;

            // create a temporary .blif file
            // Output the first three lines for .blif file
            ofstream ofs(prefix+"solution" + std::to_string(count)+".blif");
            ofs << ".model solution-" << count<< endl;
            ofs << ".inputs" << " ";
            for(int i = 0; i < variableNumber+accuracy; i++){
                if(i == variableNumber+accuracy-1){
                    ofs << "x" << i << endl;
                }
                else{
                    ofs << "x" << i << " ";
                }
            }
            ofs << ".outputs z0" << endl;

            // Convert the type solution from string to vector <int>
            while(ss >> i){
                res.push_back(i);
            }

            cout << "Assingment matrix for the output gate:" << endl;
            int t = X_unknown[gate-1][0]-1;
            // Print assignment matrix
            for(auto p = 0; p < int(AssMatrix[0].size()); p++){
                for(auto q = 0; q < int(AssMatrix.size()); q++){
                    if(res[t] > 0){
                    	AssMatrix[q][p]  = 1;
                    }
                    else{
                    	AssMatrix[q][p]  = 0;
                    }
                    t++;
                }
            }

            for(auto p = 0; p < int(AssMatrix.size()); p++){
                for(auto q = 0; q < int(AssMatrix[p].size()); q++){
                    cout << AssMatrix[p][q] << " ";
                }
                cout << endl;
            }
            cout << endl << endl;

            // circuitRes stores the results for .blif file
            vector < vector <string> > circuitRes (gate, vector <string>(4));
            vector <string> gateType{};
            for(auto i = 0; i < int(S.size()); i++){
                cout << "For gate " << variableNumber + accuracy + i << ":" << endl;
                cout << "The two fanins are: ";
                circuitRes[i][3] = std::to_string(variableNumber + accuracy + i);

                auto k = S[i][0];
                auto count = 0;
                for(auto j = 0; j < S[i].size(); j++){
                    if(res[j+k-1] > 0){
                        count++;
                        cout << j << " ";
                        if(count == 1){
                            circuitRes[i][1] = std::to_string(j);
                        }
                        else if(count == 2){
                            circuitRes[i][2] = std::to_string(j);
                        }
                    }
                }
                cout << ",the operator is: ";
                auto p = f[i][0];
                auto gateIndex = string();
                for(auto q = 0; q < int(f[i].size()); q++){
                    if(res[q+p-1] > 0){
                        gateIndex.push_back('1');
                        //cout << 1 << " ";
                    }
                    else if(res[q+p-1] < 0){
                        gateIndex.push_back('0');
                        //cout << 0 << " ";
                    }
                }
                //cout << gateIndex << " ";

                // cout << gateIndex << endl;
                auto iter = gateLibrary.find(gateIndex);
                cout << iter->second << endl;
                gateType.push_back(iter->second);
                //                circuitRes[i][0] = iter->second;

            }
            cout << endl << endl;
            count++;

            // Verify the circuit
            int index = (gate-1) * int(pow(2, accuracy+variableNumber));
            vector <int> calFeatuteVec(variableNumber+1);
            for(auto j = 0; j < int(indexOfXn_mInG.size()); j++){
                int sum = 0;
                for(auto k = 0; k < int(indexOfXn_mInG[j].size());k++){
                    int x = index+indexOfXn_mInG[j][k];
                    if(res[x] > 0){
                        sum++;
                    }
                }
                calFeatuteVec[j] = sum;
            }

            int flag = 1;
            for(auto i = 0; i < variableNumber+1; i++){
                if(calFeatuteVec[i] != featureVector[i]){
                    flag = 0;
                }
            }

            if(flag == 1){
                cout << "Verify successfully!" << endl;
            }
            else{
                cout << "Verify failed! The feature vector is: ";
                for(auto i = 0; i < variableNumber+1; i++){
                    cout << calFeatuteVec[i] << " ";
                }
            }


            //            for(auto p = 0; p < circuitRes.size(); p++){
            //                for(auto q = 0; q < circuitRes[p].size(); q++){
            //                    cout << circuitRes[p][q] << " ";
            //                }
            //                cout << endl;
            //            }

            //According to circuitRes, output the following lines for .blif file
            for(auto p = 0; p < circuitRes.size(); p++){
                ofs << ".names" << " " << circuitRes[p][0] << " ";
                if(atoi(circuitRes[p][1].c_str()) < variableNumber+accuracy){
                    ofs << "x" << circuitRes[p][1] << " ";
                }
                else if(atoi(circuitRes[p][1].c_str()) >= variableNumber+accuracy){
                    ofs << "new_n" << circuitRes[p][1] << "_" << " ";
                }

                if(atoi(circuitRes[p][2].c_str()) < variableNumber+accuracy){
                    ofs << "x" << circuitRes[p][2] << " ";
                }
                else if(atoi(circuitRes[p][2].c_str()) >= variableNumber+accuracy){
                    ofs << "new_n" << circuitRes[p][2] << "_" << " ";
                }

                if(p < circuitRes.size()-1){
                    ofs << "new_n" << circuitRes[p][3] << "_" << endl;
                }
                else{;
                    ofs << "z0" << endl;
                }

                string gate = gateType[p];
                if(gate == "and2"){
                    ofs << "11" << " " << "1" << endl;
                }
                else if(gate == "or2"){
                    ofs << "00" << " " << "0" << endl;
                }
                else if(gate == "nand2"){
                    ofs << "11" << " " << "0" << endl;
                }
                else if(gate == "nor2"){
                    ofs << "00" << " " << "1" << endl;
                }
                else if(gate == "xor2a"){
                    ofs << "01" << " " << "1" << endl;
                    ofs << "10" << " " << "1" << endl;
                }
                else if(gate == "xnor2a"){
                    ofs << "00" << " " << "1" << endl;
                    ofs << "11" << " " << "1" << endl;
                }
                else if(gate == "ab_"){
                    ofs << "10" << " " << "1" << endl;
                }
                else if(gate == "a_b"){
                    ofs << "01" << " " << "1" << endl;
                }
                else if(gate == "a+b_"){
                    ofs << "01" << " " << "0" << endl;
                }
                else if(gate == "a_+b"){
                    ofs << "10" << " " << "0" << endl;
                }
                else if(gate == "inv1"){
                    ofs << "00" << " " << "1" << endl;
                    ofs << "01" << " " << "1" << endl;
                }
                else if(gate == "inv2"){
                    ofs << "00" << " " << "1" << endl;
                    ofs << "10" << " " << "1" << endl;
                }
                else if(gate == "buf1"){
                    ofs << "10" << " " << "1" << endl;
                    ofs << "11" << " " << "1" << endl;
                }
                else if(gate == "buf2"){
                    ofs << "01" << " " << "1" << endl;
                    ofs << "11" << " " << "1" << endl;
                }

            }
            ofs << ".end" << endl;
        }

    }
    else{
        cout << "No circuit exists!" << endl;
    }

    return count;
}

void ConstructClause::MappingAfterSAT(string prefix, int solutionNum){
    char cmd[1000];
    for (auto i = 0; i < solutionNum; i++) {
        // strcpy(cmd, prefix.data());
        strcpy(cmd, "abc -c \"read ");
        strcat(cmd, prefix.data());
        strcat(cmd, "mcnc.genlib;read ");
        strcat(cmd, prefix.data());
//        strcat(cmd, "pla44/");
        if(i < 10)
        {
            strcat(cmd, "solution");
        }
        else if((i >= 10) && (i < 100)){
            strcat(cmd, "solution-0");
        }
        else if(i >= 100){
            strcat(cmd, "solution-");
        }

        strcat(cmd, std::to_string(i).data());
        strcat(cmd, ".blif;collapse;sop;fx;strash;dch;balance;map;print_stats;write ");
        strcat(cmd, prefix.data());
//        strcat(cmd, "pla44/");
        if(i < 10)
        {
            strcat(cmd, "solution");
        }
        else{
            strcat(cmd, "solution-0");
        }
        strcat(cmd, std::to_string(i).data());
        strcat(cmd, ".v;\"");
        system(cmd);
    }

}

void ConstructClause::VerifyCircuit(string prefix, int solutionNum) {
    // Transform .blif file of circuit to .pla file
    //    char cmd[1000];
    //    for (auto i = 0; i < solutionNum; i++) {
    //        strcpy(cmd, prefix.data());
    //        strcat(cmd, "abc -c \"read ");
    //        strcat(cmd, prefix.data());
    //        strcat(cmd, "mcnc.genlib;read ");
    //        strcat(cmd, prefix.data());
    //        strcat(cmd, "circuit");
    //        strcat(cmd, to_string(i).data());
    //        strcat(cmd, ".blif;collapse;write ");
    //        strcat(cmd, prefix.data());
    //        strcat(cmd, "circuit");
    //        strcat(cmd, to_string(i).data());
    //        strcat(cmd, ".pla;\"");
    //        system(cmd);
    //    }

    //    strcat(cmd, "abc -c \"read /Users/apple/Desktop/Test/Project1/Project1/mcnc.genlib;read /Users/apple/Desktop/Test/Project1/Project1/circuit1.blif;collapse;write /Users/apple/Desktop/Test/Project1/Project1/circuit1.pla;\"");

    // Transform .pla file with "-" to .pla file without "-"
    for(auto fileNum = 0; fileNum < solutionNum; fileNum++){
        string fileName = prefix + "circuit" + std::to_string(fileNum)+".pla";
        ifstream in;
        in.open(fileName);
        auto line = string();
        int pLALength = 0;

        for(int i = 0; i < 6; i++){
            getline(in, line);
            // cout << line << endl;

            if(i == 5){
                pLALength = line[line.length()-1] - 48;
            }
        }
        //        cout << "length: " << pLALength << endl;

        vector <vector <int>> resMat;  // Store .pla file without "-"
        int i = 0;
        while(i < pLALength){
            getline(in, line);
            //        cout << line << endl;

            // For each row, store the indexes of "-"
            vector <int> indexNum;
            for(auto j = 0; j < variableNumber + accuracy; j++){
                if(line[j] == '-'){
                    indexNum.push_back(j);
                }
            }

            // Convert the decimal numbers into binary values which from 0 to the size of indexNum
            vector <vector <int> > binMat;
            int num = int(indexNum.size());
            for (auto t = 0; t < pow(2, num); t++) {
                vector<int> m;
                for(int i = num-1; i>=0; i--)
                {
                    m.push_back(((t >> i) & 1));
                }
                binMat.push_back(m);
            }


            // Transform .pla file with "-" to .pla file without "-"
            for(auto k = 0; k < int(binMat.size()); k++){
                vector <int> res(variableNumber+accuracy);
                int p = 0;
                for(auto l = 0; l < variableNumber+accuracy; l++){
                    if(line[l] == '0' || line[l] == '1'){
                        res[l] = int(line[l]) - 48;
                    }
                    else{
                        res[l] = binMat[k][p];
                        p++;
                    }
                }
                auto iter = find(resMat.begin(),resMat.end(),res);
                if(iter == resMat.end()){
                    resMat.push_back(res);
                }
            }
            i++;
        }

        //    for(auto i = 0; i < int(resMat.size()); i++){
        //        for(auto j = 0; j < int(resMat[i].size()); j++){
        //            cout << resMat[i][j] << " ";
        //        }
        //        cout << endl;
        //    }
        //    cout << resMat.size() << endl;

        // Calaulate the feature vector for each circuit with .pla file
        vector <int> calFeatuteVec(variableNumber+1, 0);
        for(auto i = 0; i < int(resMat.size()); i++){
            int sum = 0;
            for(auto j = 0; j < variableNumber; j++){
                sum = sum + resMat[i][j];
            }
            for(auto k = 0; k < variableNumber+1; k++){
                if(sum == k){
                    calFeatuteVec[k]++;
                }
            }
        }

        cout << fileNum << "th PLA file: ";
        int flag = 1;
        for(auto i = 0; i < variableNumber+1; i++){
            if(calFeatuteVec[i] != featureVector[i]){
                flag = 0;
            }

        }

        if(flag == 1){
            cout << "Verify successfully!" << endl;
        }
        else{
            cout << "Verify failed! The feature vector is: ";
            for(auto i = 0; i < variableNumber+1; i++){
                cout << calFeatuteVec[i] << " ";
            }
        }
        in.close();
    }

}


