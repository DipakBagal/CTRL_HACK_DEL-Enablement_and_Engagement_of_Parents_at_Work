### Imports ###
# add imports - classes and defs
from Classifier import Predict_Support
from Classifier import load_model
import os
import sys

"""
sys.argv[1] is the input test file name given as command line arguments

"""
cwd = os.getcwd()

print('current working directory is :' ,cwd )
n = len(sys.argv)
if n > 1:
    input_fname=sys.argv[1]
    isabs = os.path.isabs(input_fname)
    if not isabs:
        path, filename = os.path.split(input_fname)
        print(path)
        input_fname = os.path.join(cwd ,input_fname )
        print(input_fname)
        dirname = os.path.dirname(__file__)
        relative_path = os.path.relpath(cwd, input_fname)
        input_fname = os.path.join(dirname, relative_path ,filename )
        print('Please provide Absolute filename path for the input file')
        
    runs = Predict_Support(input_fname)
else:
    print('No command line input file provided, predicting on sample inputfile:')
    runs = Predict_Support('test.csv')

print("Predict_Support: ", runs)