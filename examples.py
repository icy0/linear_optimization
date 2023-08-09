import numpy as np
import itertools
import time
import datetime
from exhaustive import exhaustive
from simplex import simplex_with_bland_rule
from init import init
from sv import sv
import logging

def time_exhaustive(name, lp_in_standardform):
    logging.debug("EXHAUSTIVE runs LP " + name)
    starttime = time.time_ns()
    exhaustive(lp_in_standardform)
    endtime = time.time_ns()
    duration = endtime - starttime
    logging.debug("duration EXHAUSTIVE: " + str(duration) + " nanoseconds, or " + str((duration / 1000000)) + " milliseconds\n\n")

def time_sv(name, lp_in_standardform):
    logging.debug("SV runs LP " + name)
    logging.debug("WARNING: if the LP is restricted and valid, the resulting base "
    + "and the solution might not be sorted ascendingly. the \"base index to solution component\"-relation though should be correct.")
    starttime = time.time_ns()
    sv(lp_in_standardform)
    endtime = time.time_ns()
    duration = endtime - starttime
    logging.debug("duration SV: " + str(duration) + " nanoseconds, or " + str((duration / 1000000)) + " milliseconds\n")
    logging.debug("========================================================================================")

datetime_now = datetime.datetime.now()
logging.basicConfig(format = '', 
                    filename=("all_testcases_logged_" 
                            + str(datetime_now.hour)
                            + str(datetime_now.minute)
                            + str(datetime_now.second) + "__"
                            + str(datetime_now.day) + "_"
                            + str(datetime_now.month) + "_"
                            + str(datetime_now.year) + ".txt"),
                    encoding='utf-8', level=logging.DEBUG)

# ========================================================================================
# Moebelfabrik (1.1):

A = np.array([[3, 2, 1, 2], 
              [1, 1, 1, 1], 
              [4, 3, 3 ,4]], dtype=float)

b = np.array([225, 117, 420], dtype=float)

c = np.array([19, 13, 12, 17], dtype=float)

time_exhaustive("Moebelfabrik (1.1):", (A, b, c))
time_sv("Moebelfabrik (1.1):", (A, b, c))

# ========================================================================================
# Oelraffinerie (1.3):

A = np.array([[-1, -1, -1,  -1,  0,  0,  0,  0,  0, 0, 0,  0],
              [ 0,  0,  0,   0,  0,  0,  0,  0,  1, 1, 1,  1],
              [17, -1, -6, -14,  0,  0,  0,  0,  0, 0, 0,  0],
              [ 0,  0,  0,   0, 22,  4, -1, -9,  0, 0, 0,  0],
              [ 0,  0,  0,   0,  0,  0,  0,  0, 27, 9, 4, -4],
              [ 1,  0,  0,   0,  1,  0,  0,  0,  1, 0, 0,  0],
              [ 0,  1,  0,   0,  0,  1,  0,  0,  0, 1, 0,  0],
              [ 0,  0,  1,   0,  0,  0,  1,  0,  0, 0, 1,  0],
              [ 0,  0,  0,   1,  0,  0,  0,  1,  0, 0, 0,  1]], dtype=float)

b = np.array([-15000, 
               10000, 
                   0, 
                   0, 
                   0, 
                4000, 
                5050, 
                7100, 
                4300], dtype=float)

c = np.array([9.97, 
              7.84, 
              4.64, 
              2.24, 
             11.93, 
               9.8, 
               6.6, 
               4.2, 
             14.13, 
              12.0, 
               8.8, 
               6.4], dtype=float)

time_exhaustive("Oelraffinerie (1.3):", (A, b, c))
time_sv("Oelraffinerie (1.3):", (A, b, c))

# ========================================================================================
# (1.4)

A = np.array([[1, -1]], dtype=float)
b = np.array([0], dtype=float)
c = np.array([1, -1], dtype=float)

time_exhaustive("(1.4):", (A, b, c))
time_sv("(1.4):", (A, b, c))

# ========================================================================================
# (1.7)

A = np.array([[ 1,  1, -1],
              [-1, -1,  1],
              [ 1, -2,  2]], dtype=float)
b = np.array([7, -7, 4], dtype=float)
c = np.array([2, -3, 3], dtype=float)

time_exhaustive("(1.7):", (A, b, c))
time_sv("(1.7):", (A, b, c))

# ========================================================================================
# (2.11)

A = np.array([[1, 1, 2],
              [2, 0, 3],
              [2, 1, 3]], dtype=float)
b = np.array([4, 5, 7], dtype=float)
c = np.array([3, 2, 4], dtype=float)

time_exhaustive("(2.11):", (A, b, c))
time_sv("(2.11):", (A, b, c))

# ========================================================================================
# (3.3)

A = np.array([[2, 3, 1],
              [4, 1, 2],
              [3, 4, 2]], dtype=float)
b = np.array([5, 11, 8], dtype=float)
c = np.array([5, 4, 3], dtype=float)

time_exhaustive("(3.3):", (A, b, c))
time_sv("(3.3):", (A, b, c))

# ========================================================================================
# (3.4)

A = np.array([[1, 3, 1],
              [-1, 0, 3],
              [2, -1, 2],
              [2, 3, -1]], dtype=float)
b = np.array([3, 2, 4, 2], dtype=float)
c = np.array([5, 5, 3], dtype=float)

time_exhaustive("(3.4):", (A, b, c))
time_sv("(3.4):", (A, b, c))

# ========================================================================================
# (3.17)
A = np.array([[1, 2, 3, 1],
              [1, 1, 2, 3]], dtype=float)
b = np.array([5, 3], dtype=float)
c = np.array([2,3, 5, 4], dtype=float)

time_exhaustive("(3.17):", (A, b, c))
time_sv("(3.17):", (A, b, c))

# ========================================================================================
# (3.20)

A = np.array([[ 2, -1, -2], 
              [ 2, -3,  1], 
              [-1,  1, -2]], dtype=float)
b = np.array([4, -5, -1], dtype=float)
c = np.array([1, -1, 1], dtype=float)

time_exhaustive("(3.20):", (A, b, c))
time_sv("(3.20):", (A, b, c))

# ========================================================================================
# (3.24)

A = np.array([[ 1, -1], 
              [-1, -1], 
              [ 2,  1]], dtype = float)
b = np.array([-1, -3, 2], dtype = float)
c = np.array([3, 1], dtype = float)

time_exhaustive("(3.24):", (A, b, c))
time_sv("(3.24):", (A, b, c))

# ========================================================================================
# (4.1)

A = np.array([[ 1, -1, -1, 3],
              [ 5,  1,  3, 8],
              [-1,  2,  3,-5]], dtype=float)
b = np.array([1, 55, 3], dtype=float)
c = np.array([4, 1, 5, 3], dtype=float)

time_exhaustive("(4.1):", (A, b, c))
time_sv("(4.1):", (A, b, c))

# ========================================================================================
# (4.13)

A = np.array([[ 1,  0, -4,  3,  1,  1],
              [ 5,  3,  1,  0, -5,  3],
              [ 4,  5, -3,  3, -4,  1],
              [ 0, -1,  0,  2,  1, -5],
              [-2,  1,  1,  1,  2,  2],
              [ 2, -3,  2, -1,  4,  5]], dtype=float)
b = np.array([1, 4, 4, 5, 7, 5], dtype=float)
c = np.array([4, 5, 1, 3, -5, 8], dtype=float)

time_exhaustive("(4.13):", (A, b, c))
time_sv("(4.13):", (A, b, c))