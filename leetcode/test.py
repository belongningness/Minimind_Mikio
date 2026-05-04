import numpy as np

def BCE(pred, target):
    eps = 1e-7
    pred = np.clip(pred, eps, 1-eps)

    result = -(target * np.log(pred) + (1-target) * np.log(1-pred)) 
    return np.mean(result)

pred = np.random.rand(2)
target = np.array([0,1])
print(pred)
print(BCE(pred, target))