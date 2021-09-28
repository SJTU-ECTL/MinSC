function fid = print_double_vec_to_file(fid, vec)
for ii=1:1:length(vec)
    v = vec(ii);
    fprintf(fid, '%1.4f ', v);
end
fprintf(fid, '\n');