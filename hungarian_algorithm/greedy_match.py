import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class GreedyMatch:
    def __init__(self, cost_matrix = None):
        if cost_matrix:
            self.cost_matrix = cost_matrix
        self.matching = []
        self.matching_ind = []
    def compute_profit_matrix(self, y_true, y_pred):
        """Compute a profit matrix from two sets of label vectors

        Args:
            y_true (array): Array descriptive of keys representative of the
                left vertices (i.e. input) on a bipartite graph
            y_pred (array): Array descriptive of keys representative of the
                right vertices (i.e. output) on a bipartite graph

        Returns:
            2D array: Two-dimensional array representative of the profit matrix
        """
        self.unique_true = list(np.unique(y_true))
        self.unique_pred = list(np.unique(y_pred))


        P = np.zeros([len(self.unique_true), len(self.unique_pred)])

        # For each vertex in the left
        for i,key_true in enumerate(self.unique_true):
            # For each vertex in the right
            for j,key_pred in enumerate(self.unique_pred):
                # Get the indices that match each of the vertices
                inds_true = set(np.where(y_true==key_true)[0])
                inds_pred = set(np.where(y_pred==key_pred)[0])

                # Get the indices that match both vertices
                intsec = inds_true.intersection(inds_pred)

                # Compute number of matches
                n_match = len(intsec)
                P[i,j] = n_match

        return P

    def compute_cost_matrix(self, y_true, y_pred):
        """Compute a cost matrix from two sets of label vectors, but first
            computing the profit matrix, and subtracting the matrix from the
            max value of the profit matrix

        Args:
            y_true (array): Array descriptive of keys representative of the
                left vertices (i.e. input) on a bipartite graph
            y_pred (array): Array descriptive of keys representative of the
                right vertices (i.e. output) on a bipartite graph

        Returns:
            2D array: Two-dimensional array representative of the cost matrix
        """
        self.profit_matrix = self.compute_profit_matrix(y_true, y_pred)
        return np.max(self.profit_matrix) - self.profit_matrix

    def recurse(self, profit_matrix, workers, jobs):
        # Find the current argmaxes for the given profit_matrix
        w_ind, j_ind = np.unravel_index(profit_matrix.argmax(), profit_matrix.shape)

        # Obtain the actual names and actual indices for the argmax
        match = (workers["names"][w_ind], jobs["names"][j_ind])
        match_ind = (workers["inds"][w_ind], jobs["inds"][j_ind])

        # Add it to the master matching lists
        self.matching.append(match)
        self.matching_ind.append(match_ind)


        # If there are more jobs than workers (i.e. columns
        #   than rows), add the chosen column to be deleted
        if self.n_workers < self.n_jobs:
            drop_cols = {j_ind}
            drop_rows = set({})
        # If there are more workers than jobs (i.e. rows than
        #   columns), add the chosen row to be deleted
        elif self.n_workers > self.n_jobs:
            drop_cols = set({})
            drop_rows = {w_ind}

        # Create sets for all rows and all columns
        all_cols = {k for k in range(profit_matrix.shape[1])}
        all_rows = {k for k in range(profit_matrix.shape[0])}

        # Subtract the ALL vs DROPs
        keep_cols = list(all_cols - drop_cols)
        keep_rows = list(all_rows - drop_rows)


        # Create new profit matrix
        new_profit_matrix = profit_matrix[keep_rows,:][:,keep_cols]

        # Create new jobs and new works
        new_jobs = {
            "names": np.array(jobs["names"])[keep_cols],
            "inds": np.array(jobs["inds"])[keep_cols]
        }
        new_workers = {
            "names": np.array(workers["names"])[keep_rows],
            "inds": np.array(workers["inds"])[keep_rows]
        }
        return new_profit_matrix, new_workers, new_jobs

    def fit(self, y_true, y_pred):
        self.unique_workers = list(np.unique(y_true))
        self.unique_jobs = list(np.unique(y_pred))

        self.profit_matrix = self.compute_profit_matrix(y_true, y_pred)
        profit_matrix = self.compute_profit_matrix(y_true, y_pred)
        workers = {"names": self.unique_true,
                "inds": np.arange(len(self.unique_true))}
        jobs    = {"names": self.unique_pred,
                "inds": np.arange(len(self.unique_pred))}

        self.n_workers, self.n_jobs = len(workers["names"]), len(jobs["names"])

        while 0 not in profit_matrix.shape:
            profit_matrix, workers, jobs = self.recurse(profit_matrix, workers, jobs)

        self.map_dict = {job: worker for worker, job in self.matching}
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(list(map(self.wname, self.unique_workers)))
        self.graph.add_nodes_from(list(map(self.jname, self.unique_jobs)))

        for worker, job in self.matching:
            self.graph.add_edge(self.jname(job), self.wname(worker))
    def view_graph(self, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        worker_nodes = list(map(self.wname, self.unique_workers))
        pos = nx.bipartite_layout(self.graph, worker_nodes)
        nx.draw(self.graph, with_labels=True, ax=ax, pos=pos,
                node_color="lightgray", node_size=600)

        return fig, ax
    def wname(self,name):
        return f"w_{name}"
    def jname(self,name):
        return f"j_{name}"
    def map(self, array):
        mapped_output = []
        for item in array:
            if item in self.map_dict.keys():
                label = self.map_dict[item]
            else:
                label = np.nan

            mapped_output.append(label)
        return mapped_output

