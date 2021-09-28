module \solution-0 (
  x0, x1, x2, x3,
  z0 );
  input x0, x1, x2, x3;
  output z0;
  nand3  g0(.a(x0), .b(x1), .c(x3), .O(z0));
endmodule
