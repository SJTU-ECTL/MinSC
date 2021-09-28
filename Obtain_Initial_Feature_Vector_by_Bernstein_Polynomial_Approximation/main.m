clc;
clear all;

bm_id = 7;

current_dir = '.\\';
addpath([current_dir, 'common_functions\\']);
addpath([current_dir, 'demo_target_functions\\']);

output_for_MinSC_input_filename = [current_dir, 'output_dir\\Input_files_for_MinSC\\',num2str(bm_id),'.txt'];
fid = fopen(output_for_MinSC_input_filename,'w'); 

for k = 3:10
    for n=1:k
        m = k - n;

    if(bm_id == 1)
        target_func_test = @func1;
    elseif(bm_id == 2)
        target_func_test = @func2;
    elseif(bm_id == 3)
        target_func_test = @func3;
    elseif(bm_id == 4)
        target_func_test = @func4;
    elseif(bm_id == 5)
        target_func_test = @func5;
    elseif(bm_id == 6)
        target_func_test = @func6;
    elseif(bm_id == 7)
        target_func_test = @func7;
    elseif(bm_id == 8)
        target_func_test = @func8;
    elseif(bm_id == 9)
        target_func_test = @func9;
    elseif(bm_id == 10)
        target_func_test = @func10;
    elseif(bm_id == 11)
        target_func_test = @func11;
    elseif(bm_id == 12)
        target_func_test = @func12;
    elseif(bm_id == 13)
        target_func_test = @func13;
    elseif(bm_id == 14)
        target_func_test = @func14;
    end

    degree = n;
    [coefs, obj, H, b, c] = Bern_appr(target_func_test, degree);

    Bernstein_coefficient_vec = coefs';

    feature_vector = BernCoefVec_to_featureVector_converter(Bernstein_coefficient_vec, m);

    %print FVs with different n and m
    fprintf(fid,'%g\t',n, m);
    fprintf(fid,'%g\t',feature_vector);
    fprintf(fid,'\n');

    %feature_vector = [0,7,2,2,2,1];
%     x=[0:0.001:1];
%     y = 0;
%     for i=1:n+1
%         y = y + (feature_vector(i)/(2.^m))*Bern1(x,i-1,n);
%     end
% 
%     targetApprError = target_function_user_defined(x) - y;
%     targetApprMSE =  sqrt(mse(targetApprError));
%     maxApprError = max(max(targetApprError));
%     minApprError = min(min(targetApprError));

% plot(x,target_function_user_defined(x),x,y,'LineWidth', 1)
% title('Approximation Error')
% xlim([0 1])
% %ylim([0 0.6])
% xlabel('x')
% ylabel('y')
% legend('target f(x)','appr g(x)')
    end
end

fclose(fid);

