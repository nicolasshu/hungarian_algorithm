from hungarian_algorithm import HungarianAlgorithm
C = [[7,6,2,9,2],
     [1,2,5,3,9],
     [5,3,9,6,5],
     [9,2,5,8,7],
     [2,5,3,6,1]]
C = np.array(C)
print("The cost matrix is:"); print(C)

hunger = HungarianAlgorithm()
print("Matching...")
hunger.match(C)

print("The optimal cost is:", hunger.cost)
print("The matching matrix is:"); print(hunger.match_matrix)
print("The matches are:", hunger.matching)
