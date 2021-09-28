module \solution-0 (
  x0, x1, x2, x3, x4, x5,
  z0 );
  input x0, x1, x2, x3, x4, x5;
  output z0;
  wire new_n6_, new_n7_;
  nand4  g0(.a(x1), .b(x2), .c(x3), .d(x4), .O(new_n6_));
  oai21  g1(.a(x5), .b(new_n6_), .c(x2), .O(new_n7_));
  aoi22  g2(.a(x0), .b(new_n6_), .c(x1), .d(new_n7_), .O(z0));
endmodule
