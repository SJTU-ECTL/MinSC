module \solution-0 (
  x0, x1, x2, x3, x4,
  z0 );
  input x0, x1, x2, x3, x4;
  output z0;
  wire new_n5_, new_n6_;
  oai22  g0(.a(x0), .b(x1), .c(x2), .d(x4), .O(new_n5_));
  inv1  g1(.a(new_n5_), .O(new_n6_));
  aoi21  g2(.a(x1), .b(x3), .c(new_n6_), .O(z0));
endmodule
