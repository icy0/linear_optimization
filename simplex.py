import numpy as np
import logging

def simplex_with_bland_rule(lp_in_valid_slackform):
    if(lp_in_valid_slackform == -1):                                            # if this case is triggered, INIT found the LP to be invalid and
        return -1                                                               # therefore no further calculations are necessary.

    (B, N, A, b_bar, c_bar, v) = lp_in_valid_slackform
    m = len(b_bar)
    n = len(c_bar)
    A_bar = -A                                                                  # although A_bar is initialized with A, i negate the whole matrix because
                                                                                # it is substracted from b in the linear system of equations.
    # an important note on this implementation of SIMPLEX:
    #
    # I treat the entire slackform as one big linear system of equations
    # and therefore refer to the NBVars in question with their respective
    # column index of their coefficients in both c' and A_bar.
    #
    # an example for this is made here:
    #
    #                               column no. 0   column no. 1   column no. 2
    #                                     |              |              |
    #                                     V              V              V
    #    row no. 0 ->     x_1 = 3 - (a_11 * x_4) - (a_12 * x_5) - (a_13 * x_6)
    #    row no. 1 ->     x_2 = 4 - (a_21 * x_4) - (a_22 * x_5) - (a_23 * x_6)
    #    row no. 2 ->     x_3 = 1 - (a_31 * x_4) - (a_32 * x_5) - (a_33 * x_6)
    #                     z   = 7 + (c_1  * x_4) - (c_2  * x_5) - (c_3  * x_6)
    #
    # e.g. column_index_of_x_e therefore refers to the respective column number
    # of the upper diagram in which the entering NBVar resides.
    #
    # Also important: I never sort N and B to be ascending, as i
    # do not require it to be sorted, just as i do not sort A_bar and c'.
    # e.g. if N = [1, 2, 3] and B = [4, 5, 6], and x_2 is the entering NBVar
    # and x_5 is the exiting BVar, the actual arrays will look like this afterwards:
    # N = [1, 5, 3] and B = [4, 2, 6]

    # STEP 1.1: find the NBVar with a positive coefficient in c'
    # and the lowest index to suffice the Bland-rule.
    column_index_of_x_e = -1                                                    # start with -1 to detect, if any positive coefficient was found afterwards
    lowest_index = 0xFFFFFFFF                                                   # start with maximal integer value 0xFFFFFFFF to make sure that the first index we look at is set as the new lowest index
    for index, c_bar_coeff in zip(range(n+1), c_bar):                           # index refers to the index of the current entry in N, c_bar_coeff to the corresponding coefficient in c'
        if(c_bar_coeff > 0 and N[index] < lowest_index):                        # if we find a positive coefficient in c' and the index is lower than the one before:
            column_index_of_x_e = index                                         # update the values
            lowest_index = N[index]

    # STEP 1.2: if there is no coefficient of any NBVar in c' which is > 0,
    # then stop and return the current slackform, as it already is optimal.
    if(column_index_of_x_e == -1):
        return (B, N, A, b_bar, c_bar, v)                                       # this equals the input parameter lp_in_valid_slackform

    # STEP 2.1: if all coefficients in A in the column of x_e 
    # are negative or zero, the LP is unrestricted in its range of optimal
    # solutions, which implies that there is no optimal solution.
    all_coefficients_in_A_are_negative = True
    for coeff_in_A in A[:, column_index_of_x_e]:
        if(coeff_in_A > 0):                                                     # if we find just one coefficient in A that is positive, then the LP is restricted
            all_coefficients_in_A_are_negative = False
            break

    if(all_coefficients_in_A_are_negative):
        logging.debug("SIMPLEX ended: This LP is unrestricted in its range of optimal solutions")
        return -1                                                               # we just return the LP and logging.debug out the state

    # STEP 2.2: find the leaving BVar with the minimal quotient (b'_l / a_le)
    # and the corresponding pivot row of x_l.
    row_index_of_x_l = 0                                                        # the resulting row number of our leaving BVar will be in here after the for-loop completed
    minimal_quotient = 0xFFFFFFFF                                               # we keep track of the minimal quotient
    lowest_index = 0xFFFFFFFF                                                   # again, we keep track of the lowest index to suffice the Bland-rule
    corresp_coeff = -1                                                          # we remember the coefficient of the leaving BVar as we will need it later
    for index, coeff in zip(range(0, m+1), A_bar[ : , column_index_of_x_e]):    # for each coefficient of x_e in A_bar
        if(coeff >= 0):                                                         # skip if coefficient in A_bar is positive (which would mean that it is negative in A)
            continue
        quotient = (b_bar[index] / -coeff)                                      # we calculate the quotient b'_l / a_le
        if((quotient < minimal_quotient)                                        # if this quotient is in fact smaller than the smallest of the quotients calculated before
        or ((quotient == minimal_quotient) 
         and (B[index] < B[row_index_of_x_l]))):                                # or if the quotient is exactly as small as the smallest of the quotients calculated before,
                                                                                # but has a smaller index
            minimal_quotient = quotient                                         # then update the minimal quotient
            row_index_of_x_l = index                                            # and the row of the leaving BVar
            corresp_coeff = coeff                                               # and the corresponding coefficient of the entering NBvar x_e in A_bar in that row
            lowest_index = B[index]                                             # and the index of the BVar of our new minimal quotient

    # STEP 3.1: update N and B.
    actual_entering_index = N[column_index_of_x_e]
    actual_leaving_index = B[row_index_of_x_l]
    N[column_index_of_x_e] = actual_leaving_index
    B[row_index_of_x_l] = actual_entering_index

    # STEP 3.2: pivot line has to be reordered so that x_e stands alone.
    row_of_x_l = A_bar[row_index_of_x_l]                                        # get a reference to the leaving bvar row
    row_of_x_l[column_index_of_x_e] = -1                                        # set this to the negative coefficient of the exiting BVar, which is always -1
    for index in range(0, n):                                                   # for each coefficient in A_bar in the pivot line
        row_of_x_l[index] /= -corresp_coeff                                     # divide the coefficients by the coefficient of the entering NBVar
    b_bar[row_index_of_x_l] /= -corresp_coeff                                   # do the same for the corresponding component in b'

    # STEP 3.3: x_e now has to be placed into the other lines.
    for row, row_index in zip(A_bar, range(0, m)):                              # for each row in A_bar
        if(row_index == row_index_of_x_l):                                      # the pivot line needs to be skipped, as we already reordered it before
            continue

        temp_row_of_x_l = np.copy(row_of_x_l)                                   # copy the pivot line
        temp_b_bar_entry = b_bar[row_index_of_x_l]                              # copy b' component of the pivot line
        old_coeff = row[column_index_of_x_e]                                    # find the coefficient of the entering NBVar in the currently processed row
        temp_row_of_x_l *= old_coeff                                            # multiply the copied pivot line with that coefficient
        temp_b_bar_entry *= old_coeff                                           # do the same for the corresponding component in b'

        b_bar[row_index] += temp_b_bar_entry                                    # then update the component of b' of the current row
        for entry_index in range(0, n):                                         # and do the same for the currently processed row in A_bar
            row[entry_index] += temp_row_of_x_l[entry_index]

        row[column_index_of_x_e] = temp_row_of_x_l[column_index_of_x_e]         # afterwards, set the component in A_bar which now represents
                                                                                # the coefficient of the entering NBVar to the coefficient of the pivot line
                                                                                # as this is the actual value for it.

    # STEP 3.4: x_e now has to be placed into the z-row (v + c')
    # (basically the same as above, just for the z-row).
    temp_row_of_x_l = np.copy(row_of_x_l)                                       # copy the pivot line again
    temp_b_bar_entry = b_bar[row_index_of_x_l]                                  # copy b' component of the pivot line again
    old_coeff = c_bar[column_index_of_x_e]                                      # find the coefficient of the entering NBVar in the z-row
    temp_row_of_x_l *= old_coeff                                                # multiply the copied pivot line with that coefficient
    temp_b_bar_entry *= old_coeff                                               # do the same for the corresponding component in b'

    v += temp_b_bar_entry                                                       # set v to be the multiplied b' component

    for entry_index in range(0, n):                                             # update c' by adding the multiplied pivot line coefficients to the c'-components
            c_bar[entry_index] += temp_row_of_x_l[entry_index]

    c_bar[column_index_of_x_e] = temp_row_of_x_l[column_index_of_x_e]           # afterwards, set the component in c'  which now represents
                                                                                # the coefficient of the entering NBVar to the coefficient of the pivot line
                                                                                # as this is the actual value for it.

    # STEP 4: Go back to STEP 1
    return simplex_with_bland_rule((B, N, -A_bar, b_bar, c_bar, v))