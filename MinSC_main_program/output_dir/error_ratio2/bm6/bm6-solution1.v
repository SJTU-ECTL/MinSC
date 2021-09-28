module \solution-0 (
  x0, x1, x2, x3, x4,
  z0 );
  input x0, x1, x2, x3, x4;
  output z0;
  wire new_n5_, new_n6_;
  nor3  g0(.a(x0), .b(x3), .c(x4), .O(new_n5_));
  nand2  g1(.a(x1), .b(x2), .O(new_n6_));
  nor2  g2(.a(new_n5_), .b(new_n6_), .O(z0));
endmodule
