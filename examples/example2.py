import numpy as np
from hungarian_algorithm import HungarianAlgorithm
C = [[7,6,2,9,2],
     [6,2,1,3,9],
     [5,6,8,9,5],
     [6,8,5,8,6],
     [9,5,6,4,7]]

C = np.array(C)
print("The cost matrix is:"); print(C)

hunger = HungarianAlgorithm()
print("Matching...")
hunger.match(C)

print("The optimal cost is:", hunger.cost)
print("The matching matrix is:"); print(hunger.match_matrix)
print("The matches are:", hunger.matching)
