function featureVector = BernCoefVec_to_featureVector_converter(BernCoefVec, accuracy_m)
len = length(BernCoefVec);
n = len-1;
featureVector = zeros(1,len);

for ii=1:1:n+1
    featureVector(ii) = round(BernCoefVec(ii) * nchoosek(n,ii-1) * 2^accuracy_m);
end


