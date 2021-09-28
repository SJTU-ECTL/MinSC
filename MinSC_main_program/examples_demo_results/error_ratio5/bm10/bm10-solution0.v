module \solution-0 (
  x0, x1, x2, x3,
  z0 );
  input x0, x1, x2, x3;
  output z0;
  wire new_n4_, new_n5_;
  xor2  g0(.a(x0), .b(x3), .O(new_n4_));
  nor3  g1(.a(x1), .b(x2), .c(new_n4_), .O(new_n5_));
  inv1  g2(.a(new_n5_), .O(z0));
endmodule
