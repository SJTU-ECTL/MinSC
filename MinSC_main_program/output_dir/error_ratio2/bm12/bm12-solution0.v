module \solution-0 (
  x0, x1, x2, x3, x4, x5,
  z0 );
  input x0, x1, x2, x3, x4, x5;
  output z0;
  wire new_n6_, new_n7_, new_n8_;
  oai21  g0(.a(x1), .b(x5), .c(x3), .O(new_n6_));
  oai21  g1(.a(x5), .b(new_n6_), .c(x1), .O(new_n7_));
  aoi21  g2(.a(x0), .b(new_n7_), .c(x2), .O(new_n8_));
  oai21  g3(.a(x4), .b(new_n6_), .c(new_n8_), .O(z0));
endmodule
