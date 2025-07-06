import dill
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str)

args = parser.parse_args()

assert args.file is not None, "You did not specify a file"

file_path = args.file 

with open(file_path , 'rb') as f:
    dict1 = dill.load(f)

# print(dict1)
breakpoint()
