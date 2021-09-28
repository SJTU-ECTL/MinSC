module \solution-0 (
  x0, x1, x2, x3, x4,
  z0 );
  input x0, x1, x2, x3, x4;
  output z0;
  wire new_n5_;
  oai22  g0(.a(x0), .b(x2), .c(x3), .d(x4), .O(new_n5_));
  aoi22  g1(.a(x0), .b(x2), .c(x1), .d(new_n5_), .O(z0));
endmodule
