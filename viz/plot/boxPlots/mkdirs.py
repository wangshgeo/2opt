import os

algos=["2Opt","BranchAndBoundHK","Annealing"]
cases=["burma14","ulysses16","berlin52","kroA100","ch150","gr202"]

for c in cases:
	for a in algos:
		directory = c+"_"+a
		if not os.path.exists(directory):
			os.makedirs(directory)