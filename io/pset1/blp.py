import pandas as pd
import numpy as np
from scipy.optimize import minimize
import scipy as sp
from numba import jit, njit, prange
import time
import multiprocessing as mp
import pickle
import pyblp
import os

my_working_dir = "/home/chris/files/school/ucla/second_year/fall/io/pset1/"
os.chdir(my_working_dir)
os.getcwd()

df = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
df_demog = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
demog = df_demog[["income", "income_squared", "age", "child"]].values

df["cons"] = 1

# demand variables
# linear
X1 = np.hstack((df[["prices"]], pd.get_dummies(df["product_ids"])))

# non-linear, note order is different from Nevo paper
X2 = df[["cons", "prices", "sugar", "mushy"]].values

k = 3

# price
p = df.prices.values

# number of goods per market
J = df.groupby("market_ids").sum().cons.values

# number of simulations per market
N = 20

# number of markets
T = len(J)

# find the share of the outside good
# I think this is actually the inside share; but leave it alone
df["outside"] = df["shares"].groupby(df["market_ids"]).transform("sum")

# initial delta_0 estimate: log(share) - log(share outside good)
delta_0 = np.log(df["shares"]) - np.log(df["outside"])

# markets for itj
markets = df.market_ids.values

# unique markets
marks = np.unique(df.market_ids)

# firms
firms = np.reshape(df.firm_ids.values, (-1,1))

# define a class so I can repeatedly update the delta value
class delta:
    def __init__(self, delta):
        self.delta = np.array(delta)

# initialize a delta object using the delta_0 values
d = delta(delta_0)

# set seed
np.random.seed(4096542)

# matrix of simulated values
#  number of rows = number of simulations
#  treat last column is price
# different draws for each market
# TODO figure out what k is
# THIS IS NEVO STEP 0, RIGHT?
V = np.reshape(np.random.standard_normal((k + 1) * N * T), (T * N, k + 1))



# the loops that calculate utility in a separate function so that it can be
#  run in parallel. 
@njit(parallel = True)
def util_iter(out, x2, v, p, demog, delta, sigma, pi, J, T, N):
    # first iterate over the individuals 
    for i in prange(N):
        # iterator through t and j
        tj = 0
        
        # iterate over the markets
        for t in prange(T):
            # market size of market t
            mktSize = J[t]
            
            # iterate over goods in a particular market
            for j in prange(mktSize):
                
                # calculate utility
                #  log of the numerator of equation (11) from Aviv's RA guide
                out[tj, i] = delta[tj] + \
                x2[tj, 0] * (v[N * t + i, 0] * sigma[0] +
                             np.dot(pi[0,:], demog[N * t + i,:])) + \
                x2[tj, 1] * (v[N * t + i, 1] * sigma[1] + 
                             np.dot(pi[1,:], demog[N * t + i,:])) + \
                x2[tj, 2] * (v[N * t + i, 2] * sigma[2] + 
                             np.dot(pi[2,:], demog[N * t + i,:])) + \
                x2[tj, 3] * (v[N * t + i, 3] * sigma[3] + 
                             np.dot(pi[3,:], demog[N * t + i,:]))
                tj += 1
    return out

# computes indirect utility given parameters
#  x: matrix of demand characteristics
#  v: monte carlo draws of N simulations
#  p: price vector
#  delta: guess for the mean utility
#  sigma: non-linear sigma (sigma - can think of as stdev's)
#  J: vector of number of goods per market
#  T: numer of markets
#  N: number of simulations

@jit
def compute_indirect_utility(x2, v, p, demog, delta, sigma, pi, J, T, N):
    # make sure sigma are positive
    sigma = np.abs(sigma)
    
    # output matrix
    out = np.zeros((sum(J), N))
    
    # call the iteration function to calculate utilities
    out = util_iter(out, x2, v, p, demog, delta, sigma, pi, J, T, N)
     
    return out




# computes the implied shares of goods in each market given inputs
# same inputs as above function

@jit
def compute_share(x2, v, p, demog, delta, sigma, pi, J, T, N):
    q = np.zeros((np.sum(J), N))
    
    # obtain vector of indirect utilities
    u = compute_indirect_utility(x2, v, p, demog, delta, sigma, pi, J, T, N)
    
    # exponentiate the utilities
    exp_u = np.exp(u)
    
    # pointer to first good in the market
    first_good = 0
            
    for t in range(T):
        # market size of market t
        mktSize = J[t]

        # calculate the numerator of the share eq
        numer = exp_u[first_good:first_good + mktSize,:]

        # calculate the denom of the share eq
        denom = 1 + numer.sum(axis = 0)    
          
        # calculate the quantity each indv purchases of each good in each market
        q[first_good:first_good + mktSize,:] = numer/denom
        
        first_good += mktSize
    
    # to obtain shares, assume that each simulation carries the same weight.
    # this averages each row, which is essentially computing the sahres for each
    #  good in each market. 
    s = np.matmul(q, np.repeat(1/N, N))
    
    return [q,s]


@jit
def solve_delta(s, x2, v, p, demog, delta, sigma, pi, J, T, N, tol):
    # define the tolerance variable
    eps = 10
    
    # renaming delta as delta^r
    delta_old = delta
    
    while eps > tol:
        # Aviv's step 1: obtain predicted shares and quantities
        q_s = compute_share(x2, v, p, demog, delta_old, 
                            sigma, pi, J, T, N)
        
        # extract the shares
        sigma_jt = q_s[1]
        
        # step 2: use contraction mapping to find delta
        delta_new = delta_old + np.log(s/sigma_jt)
        
        # update tolerance
        eps = np.max(np.abs(delta_new - delta_old))
        
        delta_old = delta_new.copy()
    
    return delta_old


# If you're looking at the four steps Aviv lists in his appendix, start here
# This is the objective function that we optimize the non-linear parameters over
def objective(params, s, x1, x2, v, p, demog, J, T, N, marks, markets, tol, 
              Z, weigh, firms):
    
    # optim flattens the params, so we have to redefine inside 
    sigma = params[0:4]

    alpha = sigma[-1]
    
    pi = params[4:].reshape((4,4))
    
    # number of observation JxT
    obs = np.sum(J)
    
    # force these params to be 0:
    pi[[0,2,3],1] = 0
    pi[[0,2,3],3] = 0
    pi[1,2] = 0
    
    if np.min(sigma) < 0:
        return 1e20
    
    else:
        # Aviv's step 1 & 2:
        d.delta = solve_delta(s, x2, v, p, demog, d.delta, sigma, pi, J, T, N, tol)

        # since we are using both demand and supply side variables,
        #  we want to stack the estimated delta and estimated mc
        y2 = d.delta.reshape((-1,1))

        # get linear parameters (this FOC is from Aviv's appendix)
        b = np.linalg.inv(x1.T @ Z @ weigh @ Z.T @ x1) @ (x1.T @ Z @ weigh @ Z.T @ y2)

        # Step 3: get the error term xi (also called omega)
        xi_w = y2 - x1 @ b

        g = Z.T @ xi_w / np.size(xi_w, axis = 0)

        obj = float(obs ** 2 * g.T @ weigh @ g)
        
        return obj


# obtain insruments
demand_inst_cols = [col for col in df.columns if 'demand' in col]
Z = np.hstack((df[demand_inst_cols], pd.get_dummies(df["product_ids"])))


# initial estimates for sigma
sigma = [.377, 1.848, 0.004, 0.081]

# initial estimates for pi, by row
pi1 =   [ 3.09,  0,      1.186,  0     ]
pi2 =   [16.6, -.66, 0,       11.6]
pi3 =   [-0.193,  0,      0.03,  0     ]
pi4 =   [ 1.468,  0,     -1.5,  0     ]

# optim must read all params in as a one dimensional vector
params = np.hstack((sigma, pi1, pi2, pi3, pi4))

# Recommended initial weighting matrix from Aviv's appendix
w1 = np.linalg.inv(Z.T @ Z)




t0 = time.time()

# util = compute_indirect_utility(X2, V, p, demog, d.delta, 
#                          sigma, pi, J, T, N)



# share = compute_share(X2, V, p, 
#                       demog, d.delta, sigma, pi, 
#                       J, T, N)

# delta = solve_delta(df.shares.values,
#                     X2, V, p, 
#                     demog, d.delta, sigma, pi, 
#                     J, T, N, 1e-4)

    
obj = objective(params, 
                df.shares.values, 
                X1, X2, 
                V, p, demog, 
                J, T, N, 
                marks, markets, 1e-6, 
                Z, w1, firms)    
    
time.time() - t0


res_init_wt = minimize(objective,
                      params, 
                      args = (df.shares.values, 
                            X1, X2, 
                            V, p, demog, 
                            J, T, N, 
                            marks, markets, 1e-4, 
                            Z, w1, firms), 
                      method = "Nelder-Mead")

res_init_wt.x
