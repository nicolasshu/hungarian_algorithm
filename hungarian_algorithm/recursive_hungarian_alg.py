import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from hungarian_algorithm import HungarianAlgorithm
import logging
import ipdb

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

            if 0 in cost_matrix.shape:
                break
            elif 1 not in cost_matrix.shape:
                # If the output of recursion is a 1D array, don't pad the
                #     the cost matrix. The self.recurse() will take care of it
                cost_matrix = self.pad_cost_matrix(cost_matrix)
                workers, jobs = self.pad_workers_jobs(workers, jobs)

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

        # Create a Graph
        self.map_dict = {job: worker for worker, job in self.matching}
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(list(map(self.wname, self.unique_workers)))
        self.graph.add_nodes_from(list(map(self.jname, self.unique_jobs)))
        # self.graph.add_nodes_from(self.unique_workers)
        # self.graph.add_nodes_from(self.unique_jobs)

        for worker, job in self.matching:
            self.graph.add_edge(self.jname(job), self.wname(worker))
            # self.graph.add_edge(worker, job)

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

        n_workers, n_jobs = len(workers["names"]), len(jobs["names"])

        matching = self.munkres.hungarian_alg(cost_matrix)
        matching_ind = self.munkres.matching_ind

        keep_rows, keep_cols = [], []
        drop_rows, drop_cols = [], []
        for match_names, match_inds in zip(matching, matching_ind):
            w_ind, j_ind = match_inds
            w_name, j_name = match_names
            # If one of the indices is equal or greater, it will be
            #   due to a "discard"
            # if (w_ind >= n_workers) or (j_ind >= n_jobs): continue
            # w_name, j_name = workers["names"][w_ind], jobs["names"][j_ind]
            if ("discard" not in str(w_name)) and ("discard" not in str(j_name)):
                self.matching.append((w_name, j_name))
                self.matching_ind.append((w_ind, j_ind))

                if n_workers < n_jobs:
                    drop_cols.append(j_ind)

                if n_workers > n_jobs:
                    drop_rows.append(w_ind)

            elif ("discard" in str(w_name)):
                drop_rows.append(w_ind)
            elif ("discard" in str(j_name)):
                drop_cols.append(j_ind)


        all_rows = {k for k in range(cost_matrix.shape[0])}
        all_cols = {k for k in range(cost_matrix.shape[1])}

        drop_cols = set(drop_cols)
        drop_rows = set(drop_rows)

        keep_rows = list(all_rows - drop_rows)
        keep_cols = list(all_cols - drop_cols)


        new_cost_matrix = cost_matrix[keep_rows, :][:, keep_cols]

        new_workers = {
            "names": list(np.array(workers["names"])[keep_rows]),
            "inds": list(np.array(workers["inds"])[keep_rows])
        }
        new_jobs = {
            "names": list(np.array(jobs["names"])[keep_cols]),
            "inds": list(np.array(jobs["inds"])[keep_cols])
        }

        return new_cost_matrix, new_workers, new_jobs


    def recurse_tmp(self, cost_matrix, workers, jobs):

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

        # print(cost_matrix)      # 
        n_workers = len(workers["inds"])
        n_jobs = len(jobs["inds"])

        matching = self.munkres.hungarian_alg(cost_matrix)
        matching_ind = self.munkres.matching_ind
        keep_rows, drop_cols = [], []
        keep_cols, drop_rows = [], []
        # for match_names, match_inds in zip(matching, matching_ind):
        for match_inds in matching_ind:
            w_ind, j_ind = match_inds
            if (w_ind >= len(workers["names"])) or (j_ind >= len(jobs["names"])):
                continue

            w_name, j_name = workers["names"][w_ind], jobs["names"][j_ind]
            if ("discard" not in str(w_name)) and ("discard" not in str(j_name)):
                # try:
                self.matching.append(
                    (workers["names"][w_ind], jobs["names"][j_ind]))
                self.matching_ind.append(
                    (workers["inds"][w_ind], jobs["inds"][j_ind]))
                # except:
                #     print(matching)
                #     print(matching_ind)
                #     print(np.unique([w for w,j in matching]))
                #     print(np.unique([j for w,j in matching]))
                #     print("-------------")
                #     print(w_ind, j_ind, w_name, j_name)
                #     print(workers, jobs)
                #     raise Exception()
                if n_workers < self.n_jobs:
                    # print(f"Adding {w_ind} to keep rows")
                    # print(f"Adding {j_ind} to drop_cols")
                    keep_rows.append(w_ind)
                    drop_cols.append(j_ind)
                elif n_workers > self.n_jobs:
                    # print(f"Adding {w_ind} to drop_rows")
                    # print(f"Adding {j_ind} to keep_cols")
                    drop_rows.append(w_ind)
                    keep_cols.append(j_ind)

                # print((workers["names"][w_ind], jobs["names"][j_ind]))
                # print((workers["inds"][w_ind], jobs["inds"][j_ind]))
            elif ("discard" in str(w_name)):
                drop_rows.append(w_ind)
            elif ("discard" in str(j_name)):
                drop_cols.append(j_ind)



        if n_workers < self.n_jobs:
            keep_rows = sorted(keep_rows)
            all_cols = {k for k in range(cost_matrix.shape[1])}
            keep_cols = list(all_cols - set(drop_cols))
        elif n_workers > self.n_jobs:
            keep_cols = sorted(keep_cols)
            all_rows = {k for k in range(cost_matrix.shape[0])}
            keep_rows = list(all_rows - set(drop_rows))

        print("    Rows to keep: ", keep_rows)
        print("    Cols to keep: ", keep_cols)
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

    @property
    def match_status(self):
        if self.n_workers == self.n_jobs:
            logging.info("There are as many workers as jobs")
            return 0
        elif self.n_workers > self.n_jobs:
            logging.info("There are more workers than jobs")
            return 1
        else:
            logging.info("There are less workers than jobs")
            return -1

    @property
    def shortfat(self):
        if self.match_status < 0:
            return True
        else:
            return False

    @property
    def tallskinny(self):
        if self.match_status > 0:
            return True
        else:
            return False

    @property
    def square(self):
        if self.match_status == 0:
            return True
        else:
            return False


    def wname(self,name):
        return f"w_{name}"
    def jname(self,name):
        return f"j_{name}"

    def view_graph(self, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        worker_nodes = list(map(self.wname, self.unique_workers))
        pos = nx.bipartite_layout(self.graph, worker_nodes)
        nx.draw(self.graph, with_labels=True, ax=ax, pos=pos,
                node_color="lightgray", node_size=600)

        return fig, ax

    def pad_workers_jobs(self, workers, jobs):
        n_workers = len(workers["names"])
        n_jobs = len(jobs["names"])
        if n_workers < n_jobs:
            n_row_pad = n_jobs - n_workers
            discard_array = [f"discard{str(k).zfill(n_row_pad//10)}"
                                 for k in range(n_row_pad)]
            workers["names"] += discard_array
            # workers["inds"] = np.arange(len(workers["names"]))

        if n_workers > n_jobs:
            n_col_pad = n_workers - n_jobs
            discard_array = [f"discard{str(k).zfill(n_col_pad//10)}"
                                 for k in range(n_col_pad)]
            jobs["names"] += discard_array
            # jobs["inds"] = np.arange(len(jobs["names"]))
        return workers, jobs

    def map(self, array):
        mapped_output = []
        for item in array:
            if item in self.map_dict.keys():
                label = self.map_dict[item]
            else:
                label = np.nan

            mapped_output.append(label)
        return mapped_output

