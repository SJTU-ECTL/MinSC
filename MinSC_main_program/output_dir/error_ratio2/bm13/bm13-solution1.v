module \solution-0 (
  x0, x1, x2, x3, x4, x5, x6,
  z0 );
  input x0, x1, x2, x3, x4, x5, x6;
  output z0;
  wire new_n7_, new_n8_;
  nand4  g0(.a(x0), .b(x4), .c(x5), .d(x6), .O(new_n7_));
  oai21  g1(.a(x1), .b(new_n7_), .c(x2), .O(new_n8_));
  aoi21  g2(.a(x3), .b(new_n7_), .c(new_n8_), .O(z0));
endmodule
