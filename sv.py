import numpy as np
from simplex import simplex_with_bland_rule
from init import init
from utility import standardform_to_slackform
import logging

def sv(standardform_lp):
    # STEP 1: calculate INIT
    correct_initial_slackform = init(standardform_lp)                           # pass it to INIT

    # STEP 2: calculate SIMPLEX
    result = simplex_with_bland_rule(correct_initial_slackform)                 # pass the valid slackform resulting from INIT to SIMPLEX and return the result
    
    if(result != -1):
        (B, N, A, b, c, v) = result
        logging.debug(str("This was a LP with " + str(len(b)) + " conditions and "
        + str(len(c)) + " variables. " + "Best value is " + str(v)
        + " with corresponding solution " + str(b)
        + " and corresponding base " + str(B) + "."))

    return