module \solution-0 (
  x0, x1, x2, x3,
  z0 );
  input x0, x1, x2, x3;
  output z0;
  wire new_n4_, new_n5_;
  nor2  g0(.a(x2), .b(x3), .O(new_n4_));
  aoi21  g1(.a(x0), .b(new_n4_), .c(x1), .O(new_n5_));
  aoi21  g2(.a(x0), .b(x1), .c(new_n5_), .O(z0));
endmodule
