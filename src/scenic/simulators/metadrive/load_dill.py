import dill

file_path = "checkpoint_0.pkl"

with open(file_path , 'rb') as f:
    dict1 = dill.load(f)

print(dict1)
