module \solution-0 (
  x0, x1, x2, x3, x4,
  z0 );
  input x0, x1, x2, x3, x4;
  output z0;
  wire new_n5_, new_n6_;
  nor3  g0(.a(x1), .b(x2), .c(x3), .O(new_n5_));
  inv1  g1(.a(x0), .O(new_n6_));
  oai21  g2(.a(x4), .b(new_n6_), .c(new_n5_), .O(z0));
endmodule
