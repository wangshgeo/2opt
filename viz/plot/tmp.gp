set term png
set grid
set title "Simulated Annealing (gr202.tsp)"
set xlabel "Relative Solution Quality (%)"
set ylabel "P (solve)"
set output "gr202_Annealing.png"
plot 	'gr202_Annealing/128.000000' u 1:2 t '128.000000' w lines, \
	'gr202_Annealing/16.000000' u 1:2 t '16.000000' w lines, \
	'gr202_Annealing/0.000000' u 1:2 t '0.000000' w lines, \
	'gr202_Annealing/64.000000' u 1:2 t '64.000000' w lines, \
	'gr202_Annealing/4.000000' u 1:2 t '4.000000' w lines