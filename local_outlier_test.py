#python local_outlier_test.py -df attack.csv

from sklearn import metrics
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error
import numpy as np
import argparse
import sys
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('-df', '--detect_csv_filename', dest='detect_csv_filename', help="provides name for input csv file to analyze traffic")
args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

analyze_name = args.detect_csv_filename
intervals=[]

pf2 = open(analyze_name, "r")
i=0
for line in pf2.readlines():
    try:
        ff=line.split("\t")[-1].strip('\n')
        intervals.append([float(ff),0])
        i=i+1
    except Exception as e:
        pass
pf2.close()

# retrieve the array
data = np.array(intervals)
#print data
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# summarize the shape of the training dataset
#print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
lof = LocalOutlierFactor(n_neighbors=500,contamination=0.01)
yhat = lof.fit_predict(X_train)
#print yhat
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
#print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
y_pred = model.predict(X_test)
#print y_pred
#print yhat
# evaluate predictions
#mae = mean_absolute_error(y_test, y_pred)
#print('mean absolute error: %.3f' % mae)#-->0
print "Accuracy:",metrics.accuracy_score(y_test, y_pred)
