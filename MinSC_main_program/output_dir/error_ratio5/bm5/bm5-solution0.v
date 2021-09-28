module \solution-0 (
  x0, x1, x2, x3, x4,
  z0 );
  input x0, x1, x2, x3, x4;
  output z0;
  wire new_n5_, new_n6_;
  oai21  g0(.a(x0), .b(x1), .c(x4), .O(new_n5_));
  oai21  g1(.a(x3), .b(new_n5_), .c(x2), .O(new_n6_));
  inv1  g2(.a(new_n6_), .O(z0));
endmodule
