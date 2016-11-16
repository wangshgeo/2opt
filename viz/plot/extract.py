
#For the 2 different plots
RELERROR=[0,0.5,0.6,0.7,0.8]
TIMES=[]

#
algos=["2Opt","BranchAndBoundHK","Annealing"]
cases=["ch150","gr202"]
seeds=range(1,11)
cutoff=600
opt = {}
opt["ch150"]=6528
opt["gr202"]=40160

for a in algos:
	for c in cases:
		scores=[]
		times=[]
		for s in seeds:
			file="../output/"+c+"_"+a+"_"+str(cutoff)+"_"+str(s)+".trace"
			print file
			f = open(file,"r")
			lines = f.readlines()
			print lines


