module \solution-0 (
  x0, x1, x2, x3,
  z0 );
  input x0, x1, x2, x3;
  output z0;
  wire new_n4_, new_n5_;
  oai21  g0(.a(x0), .b(x3), .c(x1), .O(new_n4_));
  inv1  g1(.a(x2), .O(new_n5_));
  nor2  g2(.a(new_n4_), .b(new_n5_), .O(z0));
endmodule
