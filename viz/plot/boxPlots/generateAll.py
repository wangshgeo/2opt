import os

os.chdir("../../")

algos=["2Opt","BranchAndBoundHK","Annealing"]
cases=["burma14","ulysses16","berlin52","kroA100","ch150","gr202"]

opt = {}
opt["burma14"]=3323
opt["ulysses16"]=6859
opt["berlin52"]=7542
opt["kroA100"]=21282
opt["ch150"]=6528
opt["gr202"]=40160

for c in cases:
	for a in algos:
		os.system("./run -p DATA/"+c+".tsp "+a+" "+str(opt[c]))