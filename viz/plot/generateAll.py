import os

os.chdir("../")

algos=["2Opt","BranchAndBoundHK","Annealing"]
cases=["ch150","gr202"]

opt = {}
opt["ch150"]=6528
opt["gr202"]=40160

for c in cases:
	for a in algos:
		os.system("./run -p DATA/"+c+".tsp "+a+" "+str(opt[c]))