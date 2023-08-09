import numpy as np
import itertools
import logging

# the function gets a LP in standardform (A, b, c) as input
# and returns:
# 
# - either the string explaining that the LP is invalid, 
#   if the LP is invalid
#
# - or a string containing information about 
#   the optimal targetfunction value, its corresponding
#   x' solution and its corresponding base, if the LP is valid
def exhaustive(lp_in_standardform):
    (A, b, c) = lp_in_standardform                                              # unpack the A-matrix and the b- and c-vectors from the standardform lp input-parameter
    n = len(c)                                                                  # n is the amount of structure variables, we can retrieve this number from the length of the vector c
    m = len(b)                                                                  # m is the amount of conditions, we can retrieve this number from the length of the vector b

    # STEP 1: Create A' and c'
    A_prime = np.hstack((A, np.identity(m)))                                    # The matrix A concatenated with the identity matrix of dimension m x m
    c_prime = np.concatenate((c, np.zeros(m)))                                  # c' is c with m-many concatenated 0s

    all_indices = np.arange(1, m+n+1, dtype=int)                                # this is a set of all possible indices, basically [n+m] = {1, ..., n+m}
    index_sets_with_len_m = find_subsets(all_indices, m)                        # this is a set of all subsets of all_indices with length m (provided by a helper function)
    bases = list()                                                              # this is the list which will hold all valid bases of the LP
    A_prime_Bs_of_bases = list()                                                # this is the list which will hold all A'_B of the corresponding valid bases.

    for index_set in index_sets_with_len_m:                                     # for each subset of all_indices with length m
        A_prime_B = np.zeros((m, m))                                            # allocate space for A'_B

        for i, j in zip(index_set, range(0, m)):                                # fill A'_B with its values, i == the current index and j == the iteration count
            for k in range(0, m):
                A_prime_B[k][j] = A_prime[k][i-1]

        if(np.linalg.det(A_prime_B) != 0):                                      # based on the determinant, determine whether B qualifies as a base or not
            bases.append(index_set)                                             # we save the base, as it is verified to be valid by it's non-zero determinant
            A_prime_Bs_of_bases.append(A_prime_B)                               # we save the previously created A_prime_B so we don't have to calculate it again

    best_targetfunc_value = 0                                                   # reserve a variable for the best targetfunction value
    best_solution = np.zeros(m+n)                                               # reserve a variable for the solution (x*) of the best base
    best_base = np.zeros(m, dtype=int)                                          # reserve a variable for the base of the best solution
    any_valid_solution_found = False

    # STEP 2: For every regular base with #B = m, 
    # calculate x' and check if x' >= 0 and if c'x' improved
    for base, A_prime_B_of_base in zip(bases, A_prime_Bs_of_bases):             # for every base and its corresponding A'_B
        x_prime_B = np.dot(np.linalg.inv(A_prime_B_of_base), b)                 # calculate x'_B = inverse_of(A'_B) * b
        x_prime = np.zeros(n+m)                                                 # allocate space for x'. We don't need to fill in the NBVars (non-base variables) afterwards, because they all are 0 anyways.

        iteration_counter = 0                                                   # this counter is for placing the values of x'_B in the correct components of x'
        outer_loop_continue = False                                             # if we find an x' with negative components, we need to skip it as it wouldn't be a valid solution. 
                                                                                # Therefore we remember with this variable, if we should skip the currently processed x'.

        for i, iteration_counter in zip(base, range(0, m)):                     # for each entry in the current base
            if(x_prime_B[iteration_counter] < 0):                               # check if the corresponding solution has a negative component, which would disqualify it as a potential best solution
                outer_loop_continue = True                                      # this is a flag for letting the outer for-loop know when to skip this iteration                
            x_prime[i-1] = x_prime_B[iteration_counter]                         # fill in x' with the corresponding x'_B components

        if(outer_loop_continue):                                                # if the flag was set before, we know that we have a solution with negative components at hand and
            continue                                                            # can therefore safely skip this solution.

        if(not outer_loop_continue and not any_valid_solution_found):
            any_valid_solution_found = True

        targetfunc_value = np.dot(c_prime, x_prime.T)                           # calculate the value of the solution. This is c' * transposed(x')
        if(targetfunc_value > best_targetfunc_value):                           # if the previously calculated value is better then the best values of the recent calculations
            best_targetfunc_value = targetfunc_value                            # update the best value,
            best_solution = x_prime                                             # the solution of the best value
            best_base = base                                                    # and the base of the best solution

    # STEP 3: if no base is regular, then the LP is invalid
    if(len(bases) == 0):                                                        # we have to account for the case, that no index set with length m is actually a valid base,
        logging.debug("EXHAUSTIVE ended: This LP is invalid, "                  # which in turn would make the entire LP invalid
        + "because there is no regular A'_B for any base.")
        return -1

    if(not any_valid_solution_found):
        logging.debug("EXHAUSTIVE ended: This LP is invalid, "                  # we also have to account for the case, that every possible solution has negative components
        + "because there is not a single valid solution.")                      # which in turn would also make the entire LP invalid
        return -1

    # STEP 4: set x* = x'[N] (basically cut all slack variables 
    # from the best x'), as we are only interested in 
    # the structure variables and not in the slack variables
    best_solution_without_slackvars = np.zeros(m)
    for index, entry in zip(range(0, m), best_base):
        best_solution_without_slackvars[index] = best_solution[entry-1]

    logging.debug(str("This was a LP with " + str(m) + " conditions and " 
    + str(n) + " variables. " + "Best value is " + str(best_targetfunc_value) 
    + " with corresponding solution " + str(best_solution_without_slackvars) 
    + " and corresponding base " + str(best_base) + "."))
    
    return



# the function gets a set of integers "set" and a length "length_of_subset" as input and
# produces every subset of the input set which has the length of the second input
# parameter "length_of_subset".
def find_subsets(set, length_of_subset):
    list_of_subsets = list(itertools.combinations(set, length_of_subset))       # retrieve all subsets as list of iterables

    amount_of_subsets = len(list_of_subsets)                                    # retrieve amount of subsets

    array_of_subsets = np.zeros(
            (amount_of_subsets, length_of_subset), dtype=int)                   # allocate space for the subsets in an array

    for i in range(0, amount_of_subsets):
        for j in range(0, length_of_subset):
            array_of_subsets[i][j] = list_of_subsets[i][j]                      # convert the list of iterable subsets to an array of subset-arrays

    return array_of_subsets                                                     # return the array of subset-arrays
