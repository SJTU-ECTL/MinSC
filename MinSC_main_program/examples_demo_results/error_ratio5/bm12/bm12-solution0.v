module \solution-0 (
  x0, x1, x2, x3,
  z0 );
  input x0, x1, x2, x3;
  output z0;
  wire new_n4_;
  aoi22  g0(.a(x0), .b(x3), .c(x1), .d(x2), .O(new_n4_));
  oai21  g1(.a(x2), .b(x3), .c(new_n4_), .O(z0));
endmodule
