import numpy as np
from hungarian_algorithm import HungarianAlgorithm

class RecursiveHungarianAlgorithm:
    def __init__(self):
        self.munkres = HungarianAlgorithm()
        self.matching = []
        self.matching_ind = []

    def fit(self, y_true, y_pred):
        self.unique_workers = list(np.unique(y_true))
        self.unique_jobs = list(np.unique(y_pred))


        self.n_workers = len(self.unique_workers)
        self.n_jobs = len(self.unique_jobs)

        self.profit_matrix = self.munkres.compute_profit_matrix(y_true, y_pred)
        self.cost_matrix = self.munkres.compute_cost_matrix(y_true, y_pred)
        cost_matrix = self.munkres.compute_cost_matrix(y_true, y_pred)
        workers = {"names": self.unique_workers,
                   "inds": np.arange(len(self.unique_workers))}
        jobs = {"names": self.unique_jobs,
                "inds": np.arange(len(self.unique_jobs))}


        while len(self.matching) < max(self.n_workers, self.n_jobs):
            cost_matrix, workers, jobs = self.recurse(cost_matrix,
                                                      workers,
                                                      jobs)

            if 1 not in cost_matrix.shape:
                # If the output of recursion is a 1D array, don't pad the
                #     the cost matrix. The self.recurse() will take care of it
                cost_matrix = self.pad_cost_matrix(cost_matrix)

        # print("----- INITIALIZING -----")
        # print("N_w   N_j   len(match)",self.n_workers, self.n_jobs, len(self.matching))
        # print("Matching:",self.matching)

        # print("----- RECURSING -----")
        # print(cost_matrix)
        # cost_matrix, workers, jobs = self.recurse(cost_matrix, workers, jobs)
        # cost_matrix = self.pad_cost_matrix(cost_matrix)
        # print(f"N_w={self.n_workers}   N_j={self.n_jobs}   len(match)={len(self.matching)}")
        # print("new workers: ", workers)
        # print("new jobs:    ", jobs)
        # print("Matching:    ",self.matching)


        # print("----- RECURSING -----")
        # print(cost_matrix)
        # cost_matrix, workers, jobs = self.recurse(cost_matrix, workers, jobs)
        # print(f"N_w={self.n_workers}   N_j={self.n_jobs}   len(match)={len(self.matching)}")
        # print("new workers: ", workers)
        # print("new jobs:    ", jobs)
        # print("Matching:    ",self.matching)

        # print("----- RECURSING -----")
        # print(cost_matrix)
        # cost_matrix, workers, jobs = self.recurse(cost_matrix, workers, jobs)
        # print(f"N_w={self.n_workers}   N_j={self.n_jobs}   len(match)={len(self.matching)}")
        # print("Matching:",self.matching)


    def recurse(self, cost_matrix, workers, jobs):

        if 1 in cost_matrix.shape:
            match_r, match_c = np.unravel_index(np.argmin(cost_matrix, axis=None), cost_matrix.shape)
            match_ind = (workers["inds"][match_r], jobs["inds"][match_c])
            match_name = (workers["names"][match_r],jobs["names"][match_c])
            self.matching.append(match_name)
            self.matching_ind.append(match_ind)

            new_cost_matrix = np.array([[]])
            new_workers = {"names": [], "inds": []}
            new_jobs = {"names": [], "inds": []}

            return new_cost_matrix, new_workers, new_jobs

        matching = self.munkres.hungarian_alg(cost_matrix)
        matching_ind = self.munkres.matching_ind
        keep_rows, drop_cols = [], []
        keep_cols, drop_rows = [], []
        for match_names, match_inds in zip(matching, matching_ind):
            w_ind, j_ind = match_inds
            w_name, j_name = match_names
            if ("discard" not in w_name) and ("discard" not in j_name):
                self.matching.append(
                    (workers["names"][w_ind], jobs["names"][j_ind]))
                self.matching_ind.append(
                    (workers["inds"][w_ind], jobs["inds"][j_ind]))
                if self.n_workers < self.n_jobs:
                    keep_rows.append(w_ind)
                    drop_cols.append(j_ind)
                elif self.n_workers > self.n_jobs:
                    drop_rows.append(w_ind)
                    keep_cols.append(j_ind)

                # print((workers["names"][w_ind], jobs["names"][j_ind]))
                # print((workers["inds"][w_ind], jobs["inds"][j_ind]))


        if self.n_workers < self.n_jobs:
            keep_rows = sorted(keep_rows)
            all_cols = {k for k in range(cost_matrix.shape[1])}
            keep_cols = list(all_cols - set(drop_cols))
        elif self.n_workers > self.n_jobs:
            keep_cols = sorted(keep_cols)
            all_rows = {k for k in range(cost_matrix.shape[0])}
            keep_rows = list(all_rows - set(drop_rows))

        # print("    Rows to keep: ", keep_rows)
        # print("    Cols to keep: ", keep_cols)
        new_workers = {
            "names": list(np.array(workers["names"])[keep_rows]),
            "inds": list(np.array(workers["inds"])[keep_rows])
        }
        new_jobs = {
            "names": list(np.array(jobs["names"])[keep_cols]),
            "inds": list(np.array(jobs["inds"])[keep_cols])
        }
        # print("    New rows:     ", new_workers)
        # print("    New cols:     ", new_jobs)
        new_cost_matrix = cost_matrix
        new_cost_matrix = new_cost_matrix[keep_rows,:]
        new_cost_matrix = new_cost_matrix[:,keep_cols]

        return new_cost_matrix, new_workers, new_jobs

    def pad_cost_matrix(self, C):
        n_rows, n_cols = C.shape
        if n_rows < n_cols:
            n_row_pad = n_cols - n_rows
            C = np.pad(C, ((0,n_row_pad), (0,0)), constant_values=np.max(C))
        if n_rows > n_cols:
            n_col_pad = n_rows - n_cols
            C = np.pad(C, ((0,0), (0,n_col_pad)), constant_values=np.max(C))
        return C

    @property
    def match_matrix(self):
        mat = np.ones(self.cost_matrix.shape) * np.nan
        for a,b in self.matching_ind:
            mat[a,b] = self.cost_matrix[a,b]
        return mat
