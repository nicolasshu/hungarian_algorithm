#%%
# https://python.plainenglish.io/hungarian-algorithm-introduction-python-implementation-93e7c0890e15
import numpy as np

class HungarianAlg:
    def __init__(self, cost_matrix=None):
        if cost_matrix:
            self.cost_matrix = cost_matrix

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
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)

        P = np.zeros([len(unique_true), len(unique_pred)])

        # For each vertex in the left
        for i,key_true in enumerate(unique_true):
            # For each vertex in the right
            for j,key_pred in enumerate(unique_pred):
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

    def hungarian_alg(self,cost_matrix):
        """Run the Hungarian Algorithm

        Args:
            cost_matrix (array): Array representative of the cost matrix, where
                the rows represent the keys of the left vertices and the columns
                represent the keys of the right vertices of the bipartite graph,
                and the element C[i,j] represents the edge weight between the
                ith left vertex and the jth right vertex.

        Returns:
            list of tuples: A list of tuples representative of the matching
                of the ith index and jth index of the row / left and column /
                right vertices of the bipartite graph
        """
        cost_matrix = np.array(cost_matrix)
        C = cost_matrix.copy()

        # Get important parameters
        n_rows = C.shape[0]

        # Subtract the minimum values of the rows, and then columns
        C = self.subtract_min_of_rows_and_cols(C)

        # Iterate until the number of lines that cross the zeros equals the
        #   dimension of the matrix
        n_lines = 0
        while n_lines < n_rows:
            # Mark the matrix, and get the matching
            #   {marked_rows & marked_cols} represent the minimum vertex cover
            matching, marked_rows, marked_cols = self.mark_matrix2(C)

            # Number of lines through rows and columns
            #     that are passing through zeros
            n_lines = len(marked_rows) + len(marked_cols)

            # If the number of lines is less than the dimension
            #     of the matrix, adjust the weights of the edges
            if n_lines < n_rows:
                C = self.adjust_weights(C, marked_rows, marked_cols)

        return matching

    def make_rowcol_false(self,zero_mat):
        """Run through the weights matrix and start to zero out the rows and
            and columns that contain zeros.
            INPUT: A boolean matrix indicating where there are zeros in the
                cost matrix (i.e. the weights matrix)

            1. Run through the rows, and obtain the row that contains the least
                number of zeros. If there are two rows that contain the least
                number of zeros, then select the one whose columns that have
                zeros have the least number of zeros. This selected row will be
                referred to as target_row
            2. Once you have a target_row, select the columns that have the
                least number of zeros.
            3. Set the values of the boolean matrix in that target row and
                target column to be False
            4. Append that coordinate to a =marked_zero= list
            5. Repeat to step 1 until there are no more zeros on the cost
               matrix (i.e. True's on the zero_mat)

        Args:
            zero_mat (array): A boolean matrix indicating where there are zeros
                in the cost matrix

        Returns:
            list of tuples: A list of tuples indicating the coordinates which
                were used to zero out the rows and columns of the cost matrix
        """
        zero_mat_copy = zero_mat.copy()

        marked_zero = []
        while True in zero_mat_copy:
            # Get row with fewest zeros
            row_zero_count = np.sum(zero_mat_copy,1)

            row_zc_nz = row_zero_count[np.nonzero(row_zero_count)]
            row_possible = np.where(row_zero_count == row_zc_nz.min())[0]

            c_count = np.sum(zero_mat_copy[row_possible,:],0)
            target_row = row_possible[np.argmin(c_count[np.nonzero(c_count)])]

            # Get cols from target_row that have zeros
            target_cols = np.where(zero_mat_copy[target_row])[0]

            # Get col with fewest zeros
            col_zero_count = np.sum(zero_mat_copy[:,target_cols],0)
            target_col_ind = np.argmin(col_zero_count[np.nonzero(col_zero_count)])
            target_col = target_cols[target_col_ind]

            # Turn that column and row into False
            zero_mat_copy[:,target_col] = False
            zero_mat_copy[target_row,:] = False

            mark = (target_row, target_col)
            marked_zero.append(mark)
        return marked_zero

    def subtract_min_of_rows_and_cols(self,C):
        """Subtract the minimum value of every row from every respective row,
            and then subtract the minimum value of every column from every
            respective column.

        Args:
            C (array): Array representative of the cost matrix, where the rows
                represent the keys of the left vertices and the columns
                represent the keys of the right vertices of the bipartite graph,
                and the element C[i,j] represents the edge weight between the
                ith left vertex and the jth right vertex.

        Returns:
            array: Reduced cost matrix, now containing zeros.
        """
        C = np.array(C)
        # STEP 1: Subtract min of each row and column
        # Subtract the min of each row
        C = C - np.expand_dims(np.min(C,1),1)

        # Subtract the min of each column
        C = C - np.expand_dims(np.min(C,0),0)
        return C
    def mark_matrix(self,C):
        """_summary_

        Args:
            C (_type_): _description_

        Returns:
            _type_: _description_
        """
        C = np.array(C)
        # STEP 2-1
        zero_mat = (C == 0)
        zero_mat_copy = zero_mat.copy()

        # RETURN MARKED_ZERO #########
        marked_zero = []
        ##############################
        while True in zero_mat_copy:
            # Get row with fewest zeros
            row_zero_count = np.sum(zero_mat_copy,1)

            row_zc_nz = row_zero_count[np.nonzero(row_zero_count)]
            row_possible = np.where(row_zero_count == row_zc_nz.min())[0]

            c_count = np.sum(zero_mat_copy[row_possible,:],0)
            target_row = row_possible[np.argmin(c_count[np.nonzero(c_count)])]

            # Get cols from target_row that have zeros
            target_cols = np.where(zero_mat_copy[target_row])[0]

            # Get col with fewest zeros
            col_zero_count = np.sum(zero_mat_copy[:,target_cols],0)
            target_col_ind = np.argmin(col_zero_count[np.nonzero(col_zero_count)])
            target_col = target_cols[target_col_ind]

            # Turn that column and row into False
            zero_mat_copy[:,target_col] = False
            zero_mat_copy[target_row,:] = False

            mark = (target_row, target_col)
            marked_zero.append(mark)

        # After the previous step, not every row/col will be marked. Only the
        #     necessary ones in order to make the zero_mat all False
        marked_zero_r = [r for (r,c) in marked_zero]
        marked_zero_c = [c for (r,c) in marked_zero]

        # STEP 2-2-1
        non_marked_row = list(set(range(C.shape[0])) - set(marked_zero_r))

        # RETURN MARKED_COLS #########
        marked_cols = []
        ##############################
        check_switch = True
        while check_switch:
            check_switch = False
            for i in range(len(non_marked_row)):
                row_array = zero_mat[non_marked_row[i], :]
                for j in range(row_array.shape[0]):
                    #step 2-2-2
                    if row_array[j] == True and j not in marked_cols:
                        #step 2-2-3
                        marked_cols.append(j)
                        check_switch = True

            for row_num, col_num in marked_zero:
                #step 2-2-4
                if row_num not in non_marked_row and col_num in marked_cols:
                    #step 2-2-5
                    non_marked_row.append(row_num)
                    check_switch = True
        # RETURN MARKED_ROWS #########
        marked_rows = list(set(range(C.shape[0])) - set(non_marked_row))
        ##############################
        return marked_zero, marked_rows, marked_cols

    def mark_matrix2(self,C):
        """Start marking the cost matrix.
            1. Create a boolean matrix based on the cost matrix, with True where
                the elements in the cost matrix equal zero, and False otherwise
            2. Identify the coordinates that can be marked, and thus turning
                nonzero values into zero by calling on `make_rowcol_false()`
                method.
            3. Identify the non-marked rows.
            4. Identify the columns that were marked
            5. Identify the rows that were marked

        Args:
            C (array): Array representative of the cost matrix, where the rows
                represent the keys of the left vertices and the columns
                represent the keys of the right vertices of the bipartite graph,
                and the element C[i,j] represents the edge weight between the
                ith left vertex and the jth right vertex.

        Returns:
            tuple: Tuple containing 3 items:
            [0] list of tuples: A list of tuples indicating the coordinates
                which were used to zero out the rows and columns of the cost 
                matrix
            [1] list: A list of indices representing the rows that were marked
            [2] list: A list of indices representing the columns that were
                marked
        """
        C = np.array(C)
        row_inds = np.arange(C.shape[0])
        # STEP 2-1
        zero_mat = (C == 0)

        # RETURN MARKED_ZERO #########
        marked_zero = self.make_rowcol_false(zero_mat)
        ##############################

        # After the previous step, not every row/col will be marked. Only the
        #     necessary ones in order to make the zero_mat all False
        marked_zero_r = [r for (r,c) in marked_zero]
        marked_zero_c = [c for (r,c) in marked_zero]

        # STEP 2-2-1
        non_marked_row = list(set(row_inds) - set(marked_zero_r))

        # RETURN MARKED_COLS #########
        marked_cols = []
        ##############################
        not_done_checking = True
        while not_done_checking:
            not_done_checking = False
            for nm_r in non_marked_row:
                for c, zero_bool in enumerate(zero_mat[nm_r]):
                    if (zero_bool == True) and (c not in marked_cols):
                        marked_cols.append(c)
                        not_done_checking = True
            for mz_r, mz_c in marked_zero:
                if (mz_r not in non_marked_row) and (mz_c in marked_cols):
                    non_marked_row.append(mz_r)
                    not_done_checking = True
        # RETURN MARKED_ROWS #########
        marked_rows = list(set(row_inds) - set(non_marked_row))
        ##############################
        return marked_zero, marked_rows, marked_cols
    def adjust_weights(self,C, marked_rows, marked_cols):
        """Given a cost matrix C, and the {marked_rows U marked_cols} create
            the minimum vertex cover, adjust the weight matrices by the
            following condition:

        delta = min_{i not in V, j not in V} C[i,j]

                  | C[i,j] - delta       (i not in V) AND (j in V)
        C[i,j] <= | C[i,j]               (i in V) XOR (j in V)
                  | C[i,j] + delta       (i in V) AND (j in V)

        Args:
            C (array): Array representative of the cost matrix, where the rows
                represent the keys of the left vertices and the columns
                represent the keys of the right vertices of the bipartite graph,
                and the element C[i,j] represents the edge weight between the
                ith left vertex and the jth right vertex.
            marked_rows (list): List of indices of the left vertices of the
                subgraph of bipartite subgraph representative of the the
                minimum vertex cover
            marked_cols (list): List of indices of the right vertices of the
                subgraph of bipartite subgraph representative of the the
                minimum vertex cover

        Returns:
            array: Adjusted cost matrix
        """
        n_rows, n_cols = C.shape

        nonzero_elements = []
        for r in range(n_rows):
            if r not in marked_rows:
                for c in range(n_cols):
                    if c not in marked_cols:
                        nonzero_elements.append(C[r,c])

        Delta = np.min(nonzero_elements)

        # Subtract Delta from edges connecting to
        #     only one vertex in Min Vertex Cover
        for r in range(n_rows):
            if r not in marked_rows:
                for c in range(n_cols):
                    if c not in marked_cols:
                        C[r,c] -= Delta

        # Add Delta to edges connected to two vertices
        #     in the Min Vertex Cover
        marked_rows, marked_cols
        for marked_row in marked_rows:
            for marked_col in marked_cols:
                C[marked_row, marked_col] += Delta

        return C
    def fit(self, y_true, y_pred):
        """Create a matching between the y_true and y_pred

        Args:
            y_true (array): Array descriptive of keys representative of the
                left vertices (i.e. input) on a bipartite graph
            y_pred (array): Array descriptive of keys representative of the
                right vertices (i.e. output) on a bipartite graph

        Returns:
            _type_: _description_
        """
        self.cost_matrix = self.compute_cost_matrix(y_true, y_pred)
        self.matching = self.hungarian_alg(self.cost_matrix)
        self.map_dict = self.mapping(self.matching)
        return self.matching

    def mapping(self, matching):
        """Create a mapping dictionary based on a matching

        Args:
            matching (list of tuples): tuples contain matches that relate a row
                to a column

        Returns:
            dict: dictionary that maps the row keys (left vertices) to the
                column keys (right vertices)
        """
        return {match_row: match_col for (match_row, match_col) in matching}

    def map(self, array):
        """Maps or assings an array to the respective matching

        Args:
            array (array): Array that will be converted to the respective match

        Returns:
            array: Matched array
        """
        return np.array([self.map_dict[item] for item in array])

    def match(self, cost_matrix = None):

        if cost_matrix is None:
            cost_matrix = np.array(self.cost_matrix)
        else:
            self.cost_matrix = np.array(cost_matrix)

        self.matching = self.hungarian_alg(self.cost_matrix)
        self.map_dict = self.mapping(self.matching)

    @property
    def cost(self):
        _cost = 0
        for a,b in self.matching:
            _cost += self.cost_matrix[a,b]
        return _cost

    @property
    def match_matrix(self):
        mat = np.zeros(self.cost_matrix.shape)
        for a,b in self.matching:
            mat[a,b] = self.cost_matrix[a,b]
        return mat


C = [[7,6,2,9,2],
     [1,2,5,3,9],
     [5,3,9,6,5],
     [9,2,5,8,7],
     [2,5,3,6,1]]
C = np.array(C)
print("The cost matrix is:"); print(C)

hunger = HungarianAlg()
print("Matching...")
hunger.match(C)

print("The optimal cost is:", hunger.cost)
print("The matching matrix is:"); print(hunger.match_matrix)
print("The matches are:", hunger.matching)

#%%
C = [[7,6,2,9,2],
     [6,2,1,3,9],
     [5,6,8,9,5],
     [6,8,5,8,6],
     [9,5,6,4,7]]

C = np.array(C)
print("The cost matrix is:"); print(C)

hunger = HungarianAlg()
print("Matching...")
hunger.match(C)

print("The optimal cost is:", hunger.cost)
print("The matching matrix is:"); print(hunger.match_matrix)
print("The matches are:", hunger.matching)

#%%
y_true = np.array([0,0,0,0,1,1,1,2,2,2,3,3])
y_pred = np.array([2,2,2,3,3,3,3,0,0,1,1,1])
print("------ Original mapping --------------")
print("y_true: ", y_true)
print("y_pred: ", y_pred)
hunger = HungarianAlg()
hunger.fit(y_true, y_pred)
y_new = hunger.map(y_pred)
print("----- y_new = hunger.map(y_pred) -----")
print("y_true: ", y_true)
print("y_new:  ", y_new)
y_new = hunger.map(y_true)
print("----- y_new = hunger.map(y_true) -----")
print("y_pred: ", y_pred)
print("y_new:  ", y_new)

#%%


