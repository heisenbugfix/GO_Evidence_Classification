import pickle as pkl

with open("negative_training_examples.pkl",'rb') as ptr:
    neg_ex = pkl.load(ptr)

with open("positive_training_examples.pkl",'rb') as ptr:
    pos_ex = pkl.load(ptr)

print("OK")