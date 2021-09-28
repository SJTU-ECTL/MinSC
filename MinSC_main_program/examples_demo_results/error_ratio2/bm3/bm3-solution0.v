module \solution-0 (
  x0, x1, x2, x3, x4,
  z0 );
  input x0, x1, x2, x3, x4;
  output z0;
  wire new_n5_;
  aoi21  g0(.a(x1), .b(x2), .c(x0), .O(new_n5_));
  aoi21  g1(.a(x3), .b(x4), .c(new_n5_), .O(z0));
endmodule
