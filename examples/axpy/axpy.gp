set terminal pdf size 30cm,20cm 
set title "Axpy Benchmark for different precisions"
set output "build/axpy/axpy.pdf"

set xlabel "number of elements"
set ylabel "computation time / msec."

f(x) = a*x
g(x) = c*x
h(x) = e*x

fit f(x) "build/axpy/axpy.data" using 1:2 via a
fit g(x) "build/axpy/axpy.data" using 1:3 via c
fit h(x) "build/axpy/axpy.data" using 1:4 via e

plot "build/axpy/axpy.data" using 1:2 with points pointtype 1 lc "blue" title "half", \
  '' using 1:3 with points pointtype 1 lc "red" title "single", \
  '' using 1:4 with points pointtype 1 lc "green" title "double", \
  f(x) lc "blue" notitle, \
  g(x) lc "red" notitle, \
  h(x) lc "green" notitle