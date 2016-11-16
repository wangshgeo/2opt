import os

algos=["2Opt","BranchAndBoundHK","Annealing"]
cases=["ch150","gr202"]

names={}
names["2Opt"]="2-Opt"
names["BranchAndBoundHK"]="Branch and Bound, Held-Karp"
names["Annealing"]="Simulated Annealing"

#xlabel="\"Run Time (s)\""
xlabel="\"Relative Solution Quality (%)\""
for c in cases:
	for a in algos:
		directory = c+"_"+a
		files = os.listdir(directory)
		if len(files)>0:
			title="\""+names[a]+" ("+c+".tsp)\""
			os.system("./plot "+directory+" "+xlabel+" "+title)
"""
xlabel="Relative Solution Quality (%)"
for c in cases:
	for a in algos:
		directory = c+"_"+a
		files = os.listdir(directory)
		if len(files)>0:
			title=names[a]+" ("+c+".tsp)"
			os.system("./plot "+directory+" "+xlabel+" "+title)
"""
