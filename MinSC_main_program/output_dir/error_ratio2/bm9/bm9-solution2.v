module \solution-0 (
  x0, x1, x2, x3, x4,
  z0 );
  input x0, x1, x2, x3, x4;
  output z0;
  wire new_n5_;
  aoi21  g0(.a(x1), .b(x3), .c(x0), .O(new_n5_));
  oai21  g1(.a(x4), .b(new_n5_), .c(x2), .O(z0));
endmodule
