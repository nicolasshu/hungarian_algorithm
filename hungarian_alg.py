#%%
# https://python.plainenglish.io/hungarian-algorithm-introduction-python-implementation-93e7c0890e15
import numpy as np

y_true = np.array([0,0,0,0,1,1,1,2,2,2,3,3])
y_pred = np.array([2,2,2,3,3,3,3,0,0,1,1,1])

C = cost_matrix(y_true, y_pred)

C = [[7,6,2,9,2],
     [6,2,1,3,9],
     [5,6,8,9,5],
     [6,8,5,8,6],
     [9,5,6,4,7]]

# C = [[7,6,2,9,2],
#      [1,2,5,3,9],
#      [5,3,9,6,5],
#      [9,2,5,8,7],
#      [2,5,3,6,1]]
C = np.array(C)
C_ = C
print(f"Cost Matrix:\n{C}")



class HungarianAlg:
    def __init__(self):
        pass

    def compute_profit_matrix(self, y_true, y_pred):
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)

        P = np.zeros([len(unique_true), len(unique_pred)])

        for i,key_true in enumerate(unique_true):
            for j,key_pred in enumerate(unique_pred):
                inds_true = set(np.where(y_true==key_true)[0])
                inds_pred = set(np.where(y_pred==key_pred)[0])
                intsec = inds_true.intersection(inds_pred)
                n_match = len(intsec)
                P[i,j] = n_match

        return P

    def compute_cost_matrix(self, y_true, y_pred):
        self.profit_matrix = self.compute_profit_matrix(y_true, y_pred)
        return np.max(self.profit_matrix) - self.profit_matrix

    def make_rowcol_false(self,zero_mat):
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
        C = np.array(C)
        # STEP 1: Subtract min of each row and column
        # Subtract the min of each row
        C = C - np.expand_dims(np.min(C,1),1)

        # Subtract the min of each col
        C = C - np.expand_dims(np.min(C,0),0)
        return C
    def mark_matrix(self,C):
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
        C = np.array(C)
        row_inds = np.arange(C.shape[0])
        # STEP 2-1
        zero_mat = (C == 0)

        # RETURN MARKED_ZERO #########
        marked_zero = make_rowcol_false(zero_mat)
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
    def adjust_weights(self,C):
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
    def hungarian_alg(self,cost_matrix):
        cost_matrix = np.array(cost_matrix)
        C = cost_matrix.copy()

        # Get important parameters
        n_rows = C.shape[0]

        # Subtract the minimum values of the rows, and then columns
        C = self.subtract_min_of_rows_and_cols(C)

        # Iterate: 
        n_lines = 0
        while n_lines < n_rows:
            # Mark the matrix, and get the matching
            matching, marked_rows, marked_cols = self.mark_matrix2(C)

            # Number of lines through rows and columns
            #     that are passing through zeros
            n_lines = len(marked_rows) + len(marked_cols)

            # If the number of lines is less than the dimension
            #     of the matrix, adjust the weights of the edges
            if n_lines < n_rows:
                C = adjust_weights(C)

        return matching

    def fit(self, y_true, y_pred):
        self.cost_matrix = self.compute_cost_matrix(y_true, y_pred)
        self.matching = self.hungarian_alg(self.cost_matrix)
        self.map_dict = self.mapping(self.matching)
        return self.matching

    def mapping(self, matching):
        return {match_row: match_col for (match_row, match_col) in matching}

    def map(self, array):
        return np.array([self.map_dict[item] for item in array])


hunger = Hungarian()
hunger.fit(y_true, y_pred)
y_true, hunger.map(y_pred)

#%%

#%%


