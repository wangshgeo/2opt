import os

algos=["2Opt","BranchAndBoundHK","Annealing"]
cases=["ch150","gr202"]

for c in cases:
	for a in algos:
		directory = c+"_"+a
		if not os.path.exists(directory):
			os.makedirs(directory)