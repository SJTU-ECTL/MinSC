module \solution-0 (
  x0, x1, x2, x3, x4, x5, x6,
  z0 );
  input x0, x1, x2, x3, x4, x5, x6;
  output z0;
  wire new_n7_, new_n8_;
  nand4  g0(.a(x0), .b(x3), .c(x4), .d(x6), .O(new_n7_));
  oai21  g1(.a(x5), .b(new_n7_), .c(x2), .O(new_n8_));
  and2  g2(.a(x1), .b(new_n8_), .O(z0));
endmodule
