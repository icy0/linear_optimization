import numpy as np
from simplex import simplex_with_bland_rule
from utility import standardform_to_slackform
import logging

def init(lp_in_standardform):
    (B, N, A_bar, b_bar, c_bar, v) = standardform_to_slackform(lp_in_standardform)
    m = len(b_bar)
    n = len(c_bar)

    # STEP 1: check for the case that b_bar already has only positive components
    b_contains_negative_components = False
    for element in b_bar:
        if(element < 0):                                                        # if we find a negative component, we set the flag b_contains_negative_components
            b_contains_negative_components = True
            break

    if(not b_contains_negative_components):                                     # if the flag b_contains_negative_components was not set, we can
        return (B, N, A_bar, b_bar, c_bar, v)                                       # return the lp in slackform

    # STEP 2: construct L_H and L_H(B_0). I directly construct L_H(B_0),
    # because we don't really need L_H.
    LH_B0_B = np.copy(B)                                                        # LH_B0_B is just a copy of B
    LH_B0_N = np.array(np.r_[N, 0.0], dtype=int)                                # LH_B0_N is the set N combined with 0, because x_0 is introduced in L_H
    LH_B0_A = np.c_[-A_bar, np.ones(m)]                                         # the L_H of L is a horizontal concatenation of the A_bar and a vector of 1s.
                                                                                # I decided to negate A_bar in here, because it is subtracted from b_bar anyways.
    LH_B0_b = np.copy(b_bar)                                                    # the LH_b is just a copy of the b_bar of L
    LH_B0_c = np.array([-1], dtype=float)                                       # The LH_B0_c is a vector with just one component, that is -1
    LH_B0_v = 0                                                                 # LH_v is 0

    # find the minimal component inside b_bar, so we know which row
    # is going to be the pivot row and what value x_0 will have.
    row_index_of_x_l = -1                                                       # after the for-loop, this variable will hold the row index of the pivot row
    min_b_component = 0xFFFFFFFF
    for component, index in zip(LH_B0_b, range(0, m)):                          # for each component in b_bar, check if it is the minimal component
        if(component < min_b_component):
            row_index_of_x_l = index
            min_b_component = component

    x_0 = -min_b_component                                                      # set x_0 to be the negated minimal component of b_bar

    # STEP 3: conduct first iteration of L_H(B_0), 
    # where x_e = x_0, x_l = x_n+k with b_k = min
    # b_i to retrieve L_H(B_1)

    temp = LH_B0_N[n]                                                           # update LH_B0_N and LH_B by swapping the x_l and x_e
    LH_B0_N[n] = LH_B0_B[row_index_of_x_l]
    LH_B0_B[row_index_of_x_l] = temp

    LH_B0_v = -x_0                                                              # update LH_b and LH_v

    corresp_coeff = -LH_B0_A[row_index_of_x_l, n]                               # collect the coefficient in LH_B0_A of x_0 in the pivot row
    pivot_row = np.negative(np.copy(LH_B0_A[row_index_of_x_l]))                 # copy the pivot row and negate it
    pivot_row[n] = 1                                                            # set the x_0 coefficient in the pivot row to 1
    LH_B0_A[row_index_of_x_l] = pivot_row                                       # update 
    LH_B0_b[row_index_of_x_l] *= corresp_coeff                                  # update the pivot row component of LH_b by multiplying it with the coefficient
    LH_B0_c = -pivot_row                                                        # since LH_B0_c was just = -x_0, we can set it to be the negated pivot row

    for index, row in zip(range(0, m), LH_B0_A):                                # now for each row in LH_B0_A
        if(index == row_index_of_x_l):                                          # that is not the pivot row
            continue
        for entry_index, entry in zip(range(0, n), row):                        # we update it by adding in the pivot row components (since the coefficient of x_0 is always 1)
            row[entry_index] += pivot_row[entry_index]
        LH_B0_b[index] += x_0                                                   # and we update LH_b accordingly aswell

    # construct L_H(B_1)
    LH_B1 = (np.copy(LH_B0_B),
             np.copy(LH_B0_N),
             np.copy(-LH_B0_A),
             np.copy(LH_B0_b),
             np.copy(LH_B0_c),
             np.copy(LH_B0_v))

    # STEP 4: calculate optimal solution of L_H with SIMPLEX(L_H(B_1))
    optimal_LH_slackform = simplex_with_bland_rule(LH_B1)
    (opt_B, opt_N, opt_A, opt_b, opt_c, opt_v) = optimal_LH_slackform
    opt_A = -opt_A

    # STEP 5: if x_0 is a NBVar in L_H(B*),
    # construct slackform L(B) of L and return L(B)
    if(0 in opt_N):
        # cut the x_0's away
        index_of_x_0_in_N = 0                                                   # first we have to find at which index x_0 is inside our optimal N, as this is not sorted
        for index, NBVar in zip(range(0, len(opt_N)), opt_N):
            if(NBVar == 0):
                index_of_x_0_in_N = index
                break 

        left_half_opt_A = opt_A[:, :index_of_x_0_in_N]                          # in here, we cut away the column, that contains the x_0's
        right_half_opt_A = opt_A[:, index_of_x_0_in_N+1:]                       # as we don't need them in the LP L.

        opt_A = np.c_[left_half_opt_A, right_half_opt_A]                        # we form the new optimal A_bar without x_0's by concatenating the halves horizontally

        left_half_of_opt_N = opt_N[:index_of_x_0_in_N]                          # we also remove the 0 from opt_N, as this again is not needed anymore in L
        right_half_of_opt_N = opt_N[index_of_x_0_in_N+1:]
        opt_N = np.hstack((left_half_of_opt_N, right_half_of_opt_N))

        new_c = np.zeros(len(opt_N))                                            # we swap the w-line with the z-line from the original LP L by introducing a placeholder
        new_v = 0                                                               # for both c_bar and v


        # now we need to find all the BVars in the c_bar-vector
        # and replace them with their corresponding rows in opt_A
        for i in range(0, len(c_bar)):                                              # for each component of c_bar
            if(N[i] in opt_B):                                                  # if the i-th coefficient in c_bar relates to a BVar, 
                                                                                # it needs to be replaced by the according row in opt_A
                coeff = c_bar[i]                                                    # we need the coefficient of that BVar in c_bar to multiply the row by it

                # we now have to find out, which row the BVar
                # relates to exactly. For that, we want to
                # find the index of N[i] in opt_B
                index_of_NBVar_in_opt_B = 0
                for opt_BVar_index in range(0, len(opt_B)):
                    if(N[i] == opt_B[opt_BVar_index]):
                        index_of_NBVar_in_opt_B = opt_BVar_index
                        break

                row = np.copy(opt_A[index_of_NBVar_in_opt_B])                   # we then copy the row of the BVar in opt_A
                row *= coeff                                                    # and multiply it by its coefficient in c_bar
                new_c += row                                                    # and lastly add it to new_c

                b_entry = opt_b[index_of_NBVar_in_opt_B]                        # and then we do the same for the corresponding b_bar entry and v
                b_entry *= coeff
                new_v += b_entry

            else:                                                               # in this case, the i-th coefficient in c_bar relates to a NBVar, 
                                                                                # which is fine but it needs to be added to new_c accordingly.
                # find out at which index N[i] is within opt_N
                index_of_NBVar_in_opt_N = 0
                for opt_NBVar_index in range(0, len(opt_N)):
                    if(N[i] == opt_N[opt_NBVar_index]):
                        index_of_NBVar_in_opt_N = opt_NBVar_index
                        break
                new_c[index_of_NBVar_in_opt_N] += c_bar[i]                          # add it to new_c


        return (opt_B, opt_N, -opt_A, opt_b, new_c, new_v)                      # we are done and therefore can return the new valid initial slackform of LP L

    # STEP 6: if x_0 is NOT a NBVar, return that LP L is invalid
    else:
        logging.debug("INIT ended: x_0 did not become a NBVar in its optimal solution, therefore this LP is invalid.")
        return -1