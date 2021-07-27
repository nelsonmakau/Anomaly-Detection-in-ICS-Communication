#python2 local_outlier.py -df 10122018-104Mega.csv -o test1.csv
#python2 local_outlier.py -df 10122018-104Mega.csv -z 1 -o test1.csv

from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs
from numpy import quantile, where, random
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse
import sys,os,time
import pandas as pd


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_csv_filename', dest='output_csv_filename', help="provides name for input csv file to analyze traffic",required=True)
parser.add_argument('-df', '--detect_csv_filename', dest='detect_csv_filename', help="provides name for input csv file to analyze traffic",required=True)
parser.add_argument('-z', '--zoom', dest='zoom', help="print in chunks of 520")
args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

zoom=None

file_name="intervals.txt"

output_csv =args.output_csv_filename
analyze_name = args.detect_csv_filename
try:
	if args.zoom !=None:
		zoom = int(args.zoom)
except Exception as e:
	raise e


intervals=[]
k=0
pf2 = open(analyze_name, "r")
for line in pf2.readlines():
    try:
        ff=line.split("\t")[-1].strip('\n')
        intervals.append([float(ff),1])
        #k=k+1
    except Exception as e:
        pass
pf2.close()

n=520
intvls = [intervals[i:i + n] for i in xrange(0, len(intervals), n)]
curr_chunk=0
outer_lofs=[]
outliers_len=0

def write_to_file(filename,cont):
	fp=open(filename,'a')
	fp.write(str(cont)+'\n')
	fp.close()

def del_file(filename):
	os.remove(filename)

def save_to_csv(txt_file,csv_file):
	df = pd.read_csv(txt_file,sep="\t",names=["interval","lol"],low_memory=False)
	df.to_csv(csv_file, sep='\t', encoding='utf-8')

for j in intvls:
	model = LocalOutlierFactor(n_neighbors=5000)
	rs=model.fit_predict(j)
	lofs_index=where(rs==-1)
	for jj in lofs_index[0]:
		outer_lofs.append(jj+(n*curr_chunk))

	if  os.path.exists(file_name):
		del_file(file_name)
	if  os.path.exists(output_csv):
		del_file(output_csv)

	print "idx\toutlier"
	outliers_len=outliers_len+len(lofs_index[0])
	for i in lofs_index[0]:
		ss= str(i+(n*curr_chunk))+"\t"+str(format(j[i][0],".4f"))+'\t'#+str(format(intervals[i+(n*curr_chunk)][0],".4f"))
		print ss

	for i in range(len(j)):
		ss=None
		if i in lofs_index[0]:
			ss= str(format(j[i][0],".4f"))+"\t"+"-1"
		else:
			ss= str(format(j[i][0],".4f"))+"\t"+"1"
		write_to_file(file_name,ss)
	curr_chunk=curr_chunk+1

	save_to_csv(file_name,output_csv)
	'''
	plt.title("Machine Learning - Local Outlier Factor")
	plt.xlabel("Packets Id")
	plt.ylabel("Packets Inter Arrival Time")
	plt.scatter( [i for i in range(len(intervals))],[i[0] for i in intervals],label="normal")
	plt.legend()
	plt.show()
	'''
	if zoom==1:
		plt.title("Machine Learning - Local Outlier Factor")
		plt.xlabel("Packets Id")
		plt.ylabel("Packets Inter Arrival Time")
		plt.scatter( [i for i in range(len(j))],[i[0] for i in j],label="Normal Packets")
		plt.scatter( [i for i in lofs_index[0]],[j[i][0] for i in lofs_index[0]],color='r',label="Outliers")
		plt.legend()
		plt.show()
		time.sleep(2)


print "Number of Outliers: ",outliers_len
#print "Final Table"
plt.title("Machine Learning - Local Outlier Factor")
plt.xlabel("Packets Id")
plt.ylabel("Packets Inter Arrival Time")
plt.scatter( [i for i in range(len(intervals))],[i[0] for i in intervals],label="Normal Packets")
plt.scatter( [i for i in outer_lofs],[intervals[i][0] for i in outer_lofs],color='r',label="Outliers")
plt.legend()
plt.show()
