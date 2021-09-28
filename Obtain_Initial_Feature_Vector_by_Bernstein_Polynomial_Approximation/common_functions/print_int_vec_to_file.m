function fid = print_int_vec_to_file(fid, vec)
for ii=1:1:length(vec)
    v = vec(ii);
    fprintf(fid, '%d ', v);
end
fprintf(fid, '\r\n');