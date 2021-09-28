module \solution-0 (
  x0, x1, x2, x3, x4, x5,
  z0 );
  input x0, x1, x2, x3, x4, x5;
  output z0;
  wire new_n6_, new_n7_, new_n8_;
  nor2  g0(.a(x1), .b(x2), .O(new_n6_));
  aoi21  g1(.a(x2), .b(x3), .c(new_n6_), .O(new_n7_));
  nand3  g2(.a(x3), .b(x4), .c(x5), .O(new_n8_));
  aoi21  g3(.a(x0), .b(new_n8_), .c(new_n7_), .O(z0));
endmodule
