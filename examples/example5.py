import numpy as np
from hungarian_algorithm import RecursiveHungarianAlgorithm

y_true = np.array(["w02","w02","w02","w03","w03","w03","w03","w00","w00","w00","w01","w03","w03","w03","w04"])
y_pred = np.array(["j00","j00","j00","j00","j01","j01","j01","j01","j01","j00","j00","j01","j01","j01","j01"])


u_y_true = np.unique(y_true)
u_y_pred = np.unique(y_pred)

print(f"In this example, we have {len(u_y_true)} workers, and {len(u_y_pred)} jobs")
print(f"\tunique(y_true) =", u_y_true)
print(f"\tunique(y_pred) =", u_y_pred)


rhunger = RecursiveHungarianAlgorithm()
print("Fitting...")
rhunger.fit(y_true, y_pred)



print("Matchings (obj.matching): \n\t", rhunger.matching)
print("Matchings Inds (obj.matching_ind): \n\t", rhunger.matching_ind)
print("Padded Cost Matrix (obj.cost_matrix):")
print(rhunger.cost_matrix)
print("Match Matrix (obj.match_matrix):")
print(rhunger.match_matrix)

rhunger.view_graph()
import matplotlib.pyplot as plt
plt.show()
