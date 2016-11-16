import sys
import os

path=sys.argv[1]
xtitle=sys.argv[2]
title=sys.argv[3]
print xtitle
print title

files = os.listdir(path)

#xtitle="Solution Quality"
ytitle="P (solve)"

plotString = "set term png\nset grid\nset title \""+title+"\"\nset xlabel \""+xtitle+"\"\nset ylabel \""+ytitle+"\"\nset output \""+path+".png\"\nplot "
for file in files:
	#print file
	plotString+="\t\'"+path+"/"+file+"\' u 1:2 t \'"+file+"\' w lines, \\\n"

plotString = plotString[:-4]

#print plotString

of = open("tmp.gp","w")
of.write(plotString)

#os.system("gnuplot tmp.gp")

#os.remove('tmp.gp')