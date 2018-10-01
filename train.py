import csv

file = open('train.py', 'r')
r = csv.reader(file)

X, y = [], []

for row in r:
	