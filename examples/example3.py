import numpy as np
from hungarian_algorithm import HungarianAlgorithm

print("------ Original mapping --------------")
y_true = np.array([0,0,0,0,1,1,1,2,2,2,3,3])
y_pred = np.array([2,2,2,3,3,3,3,0,0,1,1,1])
print(f"In this example we have {len(np.unique(y_true))} workers and {len(np.unique(y_pred))} jobs.")
print("\tunique(y_true) =", np.unique(y_true))
print("\tunique(y_pred) =", np.unique(y_pred))
print("y_true: ", y_true)
print("y_pred: ", y_pred)

print("----- y_new = hunger.map(y_pred) -----")
hunger = HungarianAlgorithm()
hunger.fit(y_true, y_pred)
y_new = hunger.map(y_pred)
print("y_true: ", y_true)
print("y_new:  ", y_new)

print("----- y_new = hunger.map(y_true) -----")
y_new = hunger.map(y_true)
print("y_pred: ", y_pred)
print("y_new:  ", y_new)
