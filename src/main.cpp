#include "clause.hpp"
#include "target_functions.h"
#include "node.h"
#include <chrono>


int main(int argc, char* argv[]){
    std::ios::sync_with_stdio(false);  // Speed IO
    std::cin.tie(0);

    int bm_id_under_test = 5;
    double approx_error_bound_ratio = 0.05;

    double(*fp_target_func)(double);

    // assign target function
	switch(bm_id_under_test){
		case 1:
			fp_target_func = &bm1_target_func;
			break;
		case 2:
			fp_target_func = &bm2_target_func;
			break;
		case 3:
			fp_target_func = &bm3_target_func;
			break;
		case 4:
			fp_target_func = &bm4_target_func;
			break;
		case 5:
			fp_target_func = &bm5_target_func;
			break;
		case 6:
			fp_target_func = &bm6_target_func;
			break;
		case 7:
			fp_target_func = &bm7_target_func;
			break;
		case 8:
			fp_target_func = &bm8_target_func;
			break;
		case 9:
			fp_target_func = &bm9_target_func;
			break;
		case 10:
			fp_target_func = &bm10_target_func;
			break;
		case 11:
			fp_target_func = &bm11_target_func;
			break;
		case 12:
			fp_target_func = &bm12_target_func;
			break;
		case 13:
			fp_target_func = &bm13_target_func;
			break;
		case 14:
			fp_target_func = &bm14_target_func;
			break;
	}

	string prefix = "./";
	int startIndex = 3;

	int maxIterationPerRun = 1;
	// The number of gates in gate library
	int gateKind = 86;
	int maxGate = 10;
	int maxArea = 100;

	vector <int> MDV{};
	MDV.push_back(0);
	MDV.push_back(0);
	MDV.push_back(2);
	MDV.push_back(2);
	MDV.push_back(2);
	MDV.push_back(3);
	MDV.push_back(3);
	MDV.push_back(8);

   // get L2norm of the target function
   double L2norm_of_target_function = get_L2_norm_of_target_function(fp_target_func, NORM_EVAL_POINT_NUM);
   cout << "L2norm_of_target_function = " << L2norm_of_target_function << endl;

//   // set the error bound according to target function L2norm
   double approx_error_bound = L2norm_of_target_function * approx_error_bound_ratio;
   cout << "approx_error_bound = " << approx_error_bound << endl;

   auto start = chrono::system_clock::now();
   ApprInfoSet lowerDegreeApprSet;
   lowerDegreeApprSet = findAFV(prefix, approx_error_bound, fp_target_func, MDV, startIndex, bm_id_under_test, approx_error_bound_ratio);

   ApprInfo ASCPSet;
   for(auto i = 0; i < lowerDegreeApprSet.size(); i++){
	   if(!lowerDegreeApprSet[i].empty()){
		   for(auto j = 0; j < lowerDegreeApprSet[i].size(); j++){
			   SingleApprInfo appr = lowerDegreeApprSet[i][j];
			   ASCPSet.push_back(appr);
		   }
	   }
   }

   for(int j = 0; j < ASCPSet.size(); j++){
	   cout << "ASCP " << j << ": ";
	   cout << "(" << ASCPSet[j].first[0] << "," << ASCPSet[j].first[1] << "), ";
	   for (auto a = 0; a < ASCPSet[j].second.size(); a++) {
           cout << ASCPSet[j].second[a] << " ";
	   }
	   cout << endl;
   }

    vector <int> maxLevel;
   	vector <int> coarseGrain1;
   	vector <int> coarseGrain2;
   	vector <int> cubeStartIndex;
   	vector < vector <int> > toBeAssignedCV1;
   	vector < vector <int> > toBeAssignedCV2;

   	if(approx_error_bound_ratio == 0.02){
   		int mgs2[5] = {5, 8, 11, 12, 13};
   		vector <int> mgs_vec2(mgs2, mgs2+5);

   		auto iter = find(mgs_vec2.begin(), mgs_vec2.end(), bm_id_under_test);
   		if(iter != mgs_vec2.end()){
   			char cmd[100];

   			int vaule = approx_error_bound_ratio * 100;
   			strcpy(cmd, "../DCV_MGS/error_ratio");
   			strcat(cmd, std::to_string(vaule).data());
   			strcat(cmd, "/");
   			strcat(cmd, std::to_string(bm_id_under_test).data());
   			strcat(cmd, ".txt");
   			ifstream input(cmd);

   			int temp2;
   			auto line = string();
   			stringstream ss;
   			ss.str(string());
   			ss.clear();
   			getline(input, line); // first line
   			ss << line;
   			while (ss >> temp2)
   			{
   				maxLevel.push_back(temp2);
   			}

   			ss.str(string());
   			ss.clear();
   			getline(input, line); // second line
   			ss << line;
   			while (ss >> temp2)
   			{
   				coarseGrain1.push_back(temp2);
   			}

   			ss.str(string());
   			ss.clear();
   			getline(input, line); // second line
   			ss << line;
   			while (ss >> temp2)
   			{
   				coarseGrain2.push_back(temp2);
   			}

   			ss.str(string());
   			ss.clear();
   			getline(input, line); //
   			ss << line;
   			while (ss >> temp2)
   			{
   				cubeStartIndex.push_back(temp2);
   			}


   			vector <int> tmp;
   			for(auto i = 0; i < maxLevel.size(); i++){
   				tmp.clear();
   				int temp1;
   				ss.str(string());
   				ss.clear();
   				getline(input, line); // second line
   				ss << line;
   				while (ss >> temp1)
   				{
   					tmp.push_back(temp1);
   				}
   				toBeAssignedCV1.push_back(tmp);
   			}

   			for(auto i = 0; i < maxLevel.size(); i++){
   				tmp.clear();
   				int temp1;
   				ss.str(string());
   				ss.clear();
   				getline(input, line); // second line
   				ss << line;
   				while (ss >> temp1)
   				{
   					tmp.push_back(temp1);
   				}
   				toBeAssignedCV2.push_back(tmp);
   			}
   		}
   		else{
   			maxLevel = vector <int>(int(ASCPSet.size()), 0);
   			coarseGrain1 = vector <int>(int(ASCPSet.size()), 1);
   			coarseGrain2 = vector <int>(int(ASCPSet.size()), 1);
   			cubeStartIndex = vector <int>(int(ASCPSet.size()), 0);

   			for(auto i = 0; i < ASCPSet.size(); i++){
   				vector <int> tmp(int(ASCPSet[i].second.size()), 0);
   				toBeAssignedCV1.push_back(tmp);
   				toBeAssignedCV2.push_back(tmp);
   			}

   		}
   	}
   	else if(approx_error_bound_ratio == 0.05){
   		int mgs5[1] = {14};
   		vector <int> mgs_vec5(mgs5, mgs5+1);

   		auto iter = find(mgs_vec5.begin(), mgs_vec5.end(), bm_id_under_test);
		if(iter != mgs_vec5.end()){
			char cmd[100];

			int vaule = approx_error_bound_ratio * 100;
			strcpy(cmd, "../DCV_MGS/error_ratio");
			strcat(cmd, std::to_string(vaule).data());
			strcat(cmd, "/");
			strcat(cmd, std::to_string(bm_id_under_test).data());
			strcat(cmd, ".txt");
			ifstream input(cmd);

			int temp2;
			auto line = string();
			stringstream ss;
			ss.str(string());
			ss.clear();
			getline(input, line); // first line
			ss << line;
			while (ss >> temp2)
			{
				maxLevel.push_back(temp2);
			}

			ss.str(string());
			ss.clear();
			getline(input, line); // second line
			ss << line;
			while (ss >> temp2)
			{
				coarseGrain1.push_back(temp2);
			}

			ss.str(string());
			ss.clear();
			getline(input, line); // second line
			ss << line;
			while (ss >> temp2)
			{
				coarseGrain2.push_back(temp2);
			}

			ss.str(string());
			ss.clear();
			getline(input, line); //
			ss << line;
			while (ss >> temp2)
			{
				cubeStartIndex.push_back(temp2);
			}


			vector <int> tmp;
			for(auto i = 0; i < maxLevel.size(); i++){
				tmp.clear();
				int temp1;
				ss.str(string());
				ss.clear();
				getline(input, line); // second line
				ss << line;
				while (ss >> temp1)
				{
					tmp.push_back(temp1);
				}
				toBeAssignedCV1.push_back(tmp);
			}

			for(auto i = 0; i < maxLevel.size(); i++){
				tmp.clear();
				int temp1;
				ss.str(string());
				ss.clear();
				getline(input, line); // second line
				ss << line;
				while (ss >> temp1)
				{
					tmp.push_back(temp1);
				}
				toBeAssignedCV2.push_back(tmp);
			}
		}
		else{
			maxLevel = vector <int>(int(ASCPSet.size()), 0);
			coarseGrain1 = vector <int>(int(ASCPSet.size()), 1);
			coarseGrain2 = vector <int>(int(ASCPSet.size()), 1);
			cubeStartIndex = vector <int>(int(ASCPSet.size()), 0);

			for(auto i = 0; i < ASCPSet.size(); i++){
				vector <int> tmp(int(ASCPSet[i].second.size()), 0);
				toBeAssignedCV1.push_back(tmp);
				toBeAssignedCV2.push_back(tmp);
			}

		}
   	}


   const int N = 4; //number of fanins for the operator

   unordered_map <string, string> gateLibrary;
   vector <vector <int> > gateTruthTable{};
   vector <double> gateArea{};

   ifstream in(prefix+"gate.txt");
   auto line1 = string();
   double area = 0;
   auto str1 = string();
   auto str2 = string();
   stringstream ss1;

   int num = 0;
   while(getline(in, line1)){
	num++;
	if(num > gateKind){
		break;
	}

	ss1.str(string());
	ss1.clear();
	ss1 << line1;
	ss1 >> str1 >> area >> str2;
	gateLibrary.insert(pair <string, string> (str2, str1));
	gateArea.push_back(area);

	vector <int> tmp;
	tmp.clear();

	for(auto j = 0; j < str2.size(); j++){
		tmp.push_back(str2[j]-'0');
	}

	gateTruthTable.push_back(tmp);
   }

   int gateCate = 0;  // The number of gate indexes need to be added for each gate
   for(auto i = 0; i < gateArea.size(); i++){
	   gateCate = gateCate + gateArea[i];
   }


  vector <int> areaSet(int(ASCPSet.size()), 100);
  vector <int> gateNumSet(int(ASCPSet.size()), 100);

for(int v = 0; v < ASCPSet.size(); v++){
    // first line: number of variables n and accuracy m
    // second line: gate r
    // following line:problem vector
	int currentMinGateNum;
	if(v == 0){
		currentMinGateNum = maxGate;
	}
	else{
		vector <int> subGateNumSet(gateNumSet.begin(), gateNumSet.begin()+v);
		currentMinGateNum = *min_element(subGateNumSet.begin(), subGateNumSet.end());
	}
	cout << endl;

	cout << "For ASCP " << v << ":" << endl;
    int variableNumber = ASCPSet[v].first[0];
    int accuracy = ASCPSet[v].first[1];
    vector<int> featureVector = ASCPSet[v].second;

    if(variableNumber + accuracy < N){
    	int tmp = N - (variableNumber + accuracy);
    	accuracy = accuracy + tmp;
    	for(auto t = 0; t < featureVector.size(); t++){
    		featureVector[t]  = featureVector[t] * pow(2, tmp);
    	}
    }

    int gate;
    int result = 0;

    int minGateNum ;
    double A0;

    auto caseCount = 0;
	vector <int> degrees{};
	degrees.push_back(variableNumber);
	auto solutionTree = SolutionTree(featureVector, degrees, accuracy, caseCount);

	vector<Node> _nodeVector;
	_nodeVector = solutionTree.ProcessTree(maxLevel[v]);

    for(auto gate = 1; gate < maxGate; gate++){
    	if( gate > currentMinGateNum){
    		A0 = maxArea;
    		break;
    	}
    	cout << "gate Number:" << gate << endl;
		context c;
		solver s(c);
    
		auto solutionClause = ConstructClause(variableNumber, accuracy, gate, gateCate, gateKind, N, featureVector, gateTruthTable, c, gateLibrary);

		MatInfo IndexMatrix = solutionClause.PrepareForConstruct2ndClause();

	    for(auto cubeNum = cubeStartIndex[v]; cubeNum < cubeStartIndex[v] + 1; cubeNum++){
	        // Determine largestCubeVector which is on the first level
//	        cout << "CubeNum: " << cubeNum << endl;
	        vector <int> largestCubeVector(int(featureVector.size()), 0);
	        for(auto i = 0; i < int(featureVector.size()); i++){
	            largestCubeVector[i] = featureVector[i] - _nodeVector[cubeNum]._remainingProblemVector[i];
	        }

//	        cout << "The largest cube vector: ";
//	        for(auto i = 0; i < int(largestCubeVector.size()); i++){
//	            cout << largestCubeVector[i] << " ";
//	        }
//	        cout << endl;

	        solutionClause.AssMatrix.clear();
	        vector <int> temp;
	        for(auto i = 0; i < _nodeVector[cubeNum]._assignedAssMat.size(); i++){
	            temp.clear();

	            for(auto j = 0; j < _nodeVector[cubeNum]._assignedAssMat[0].size(); j++){
	                if(_nodeVector[cubeNum]._assignedAssMat[i][j]-'0' == 0){
	                    temp.push_back(-1);
	                }
	                else{
	                    temp.push_back(_nodeVector[cubeNum]._assignedAssMat[i][j]-'0');
	                }
	            }
	            solutionClause.AssMatrix.push_back(temp);
	        }


			vector <vector <IndexRelation> > AllVarRelation = solutionClause.AddAdditionalVariable(c, IndexMatrix, largestCubeVector, featureVector, toBeAssignedCV1[v], coarseGrain1[v], toBeAssignedCV2[v], coarseGrain2[v]);

			
			ofstream outputfile;
			outputfile.open(prefix+"test53.cnf");

			solutionClause.ConstructMainClause(outputfile, s);

			solutionClause.Construct1stClause(outputfile, c, s);

			solutionClause.ConstructFVClauseWithoutESPRESSO(prefix, outputfile, featureVector, largestCubeVector, toBeAssignedCV1[v], AllVarRelation, coarseGrain1[v], IndexMatrix, toBeAssignedCV2[v], coarseGrain2[v], c, s);

			solutionClause.GateIndexConstraintZ3(outputfile, gateKind, c, s);

			result = solutionClause.SMTSolver(c, s, maxIterationPerRun, IndexMatrix.second, gateArea);
		}
		if(result == 1){
			model m = s.get_model();
			minGateNum = gate;
			gateNumSet[v] = minGateNum;
			A0 = solutionClause.PrintVerifyCircuitZ3(m, IndexMatrix.second, gateArea, v, areaSet);
			break;
		}
    }

//    int flag = 0;
    for(A0 = A0 - 1; A0 > 0;){
    	if(A0 ==  maxArea - 1){
    		cout << "Filter out this ASCP!" << endl;
    		cout << endl << endl;
    		break;
    	}
    	cout << endl << endl;
    	cout << "Circuit area:" << A0 << endl;
		context c;
		solver s(c);

		auto solutionClause = ConstructClause(variableNumber, accuracy, minGateNum, gateCate, gateKind, N, featureVector, gateTruthTable, c, gateLibrary);
		MatInfo IndexMatrix = solutionClause.PrepareForConstruct2ndClause();

		for(auto cubeNum = cubeStartIndex[v]; cubeNum < cubeStartIndex[v] + 1; cubeNum++){
			vector <int> largestCubeVector(int(featureVector.size()), 0);

			for(auto i = 0; i < int(featureVector.size()); i++){
				largestCubeVector[i] = featureVector[i] - _nodeVector[cubeNum]._remainingProblemVector[i];
			}
    	
//			cout << "The largest cube vector: ";
//			for(auto i = 0; i < int(largestCubeVector.size()); i++){
//				cout << largestCubeVector[i] << " ";
//			}
//			cout << endl;

			solutionClause.AssMatrix.clear();
			vector <int> temp;
			for(auto i = 0; i < _nodeVector[cubeNum]._assignedAssMat.size(); i++){
				temp.clear();

				for(auto j = 0; j < _nodeVector[cubeNum]._assignedAssMat[0].size(); j++){
					if(_nodeVector[cubeNum]._assignedAssMat[i][j]-'0' == 0){
						temp.push_back(-1);
					}
					else{
						temp.push_back(_nodeVector[cubeNum]._assignedAssMat[i][j]-'0');
					}
				}
				solutionClause.AssMatrix.push_back(temp);
			}

    		vector <vector <IndexRelation> > AllVarRelation = solutionClause.AddAdditionalVariable(c, IndexMatrix, largestCubeVector, featureVector, toBeAssignedCV1[v], coarseGrain1[v], toBeAssignedCV2[v], coarseGrain2[v]);

			ofstream outputfile;
			outputfile.open(prefix+"test53.cnf");

			solutionClause.ConstructMainClause(outputfile, s);

			solutionClause.Construct1stClause(outputfile, c, s);


			solutionClause.ConstructFVClauseWithoutESPRESSO(prefix, outputfile, featureVector, largestCubeVector, toBeAssignedCV1[v], AllVarRelation, coarseGrain1[v], IndexMatrix, toBeAssignedCV2[v], coarseGrain2[v], c, s);

			solutionClause.GateIndexConstraintZ3(outputfile, gateKind, c, s);
			solutionClause.areaConstraintZ3(outputfile, gateKind, gateArea, A0, c, s);

			result = solutionClause.SMTSolver(c, s, maxIterationPerRun, IndexMatrix.second, gateArea);
		}

		if(result == 1){
			model m1 = s.get_model();
			int tmpArea = solutionClause.PrintVerifyCircuitZ3(m1, IndexMatrix.second, gateArea, v, areaSet);
			A0 = tmpArea - 1;
		}
		if(result == 0){
			cout << "Find optimal area!" << endl << endl;
//			flag = 1;
			break;
		}
    //    if(flag == 1){
    // 	  break;
    //     }
    }
  }

	auto minPos = min_element(areaSet.begin(), areaSet.end());
	cout << "The optimal ASCP is: ASCP" << minPos - areaSet.begin();
	cout << ", and the optimal area is: " << *minPos << endl << endl;

	cout << "Total running time:" << endl;
	auto end1 = chrono::system_clock::now();
	auto duration1 = chrono::duration_cast<std::chrono::milliseconds>(end1 - start);
	auto milliseconds1 = duration1.count();
	cout << "milliseconds: " << milliseconds1 << endl;
	auto hours1 = milliseconds1 / 3600000;
	milliseconds1 -= hours1 * 3600000;
	auto minitues1 = milliseconds1 / 60000;
	milliseconds1 -= minitues1 * 60000;
	auto seconds1 = milliseconds1 / 1000;
	milliseconds1 -= seconds1 * 1000;
	cout << "Time used: ";
	cout << hours1 << ":";
	cout.fill('0');
	cout.width(2);
	cout << minitues1 << ":";
	cout.fill('0');
	cout.width(2);
	cout << seconds1 << ".";
	cout.fill('0');
	cout.width(3);
	cout << milliseconds1 << endl;
	cout << endl;

	return 0;
}

