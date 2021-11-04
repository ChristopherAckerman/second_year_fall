!find / -iname 'libdevice'
!find / -iname 'libnvvm.so'

#Add two libraries to numba environment variables:
import os
from scipy.optimize import minimize
os.environ['NUMBAPRO_LIBDEVICE'] = "/usr/local/cuda-10.0/nvvm/libdevice"
os.environ['NUMBAPRO_NVVM'] = "/usr/local/cuda-10.0/nvvm/lib64/libnvvm.so"

#install linear and pyBLP models
!pip install linearmodels
!pip install pyBLP

"""note: must restart runtime"""

# Import Packages

import pandas as pd                    # for data handling
import numpy as np                     # for numerical methods and data structures
import matplotlib.pyplot as plt        # for plotting
import seaborn as sea                  # advanced plotting
import patsy                           # provides a syntax for specifying models
import linearmodels.iv as iv           # provides IV statistical modeling
import statsmodels.api as sm           # provides statistical models like ols, gmm, anova, etc...
import statsmodels.formula.api as smf  # provides a way to directly spec models from formulas
import pyblp                           # for BLP
import time
from numba import njit, prange, cuda   # for acceleration of functions
from statsmodels.iolib.summary2 import summary_col

# Login to drive

from google.colab import auth
auth.authenticate_user()

import gspread
from oauth2client.client import GoogleCredentials

gc = gspread.authorize(GoogleCredentials.get_application_default())

# Download data

otc_data = gc.open_by_url('https://docs.google.com/spreadsheets/d/1YP2uhQ-l4MF2HzjesLa0qcZG0olk-Opqdb1Ew9Z4qPo/edit#gid=0')
otc_data = otc_data.worksheet('Sheet1')
otc_dataDf = pd.DataFrame(otc_data.get_all_records())
print(otc_dataDf.head())

otc_demographics = gc.open_by_url('https://docs.google.com/spreadsheets/d/1RL7sbL4YJs9C6hDiDlVZSDb5SWpN6EHATmbHZQL9Crc/edit#gid=0')
otc_demographics = otc_demographics.worksheet('Sheet1')
otc_demographicsDf = pd.DataFrame(otc_demographics.get_all_records())
print(otc_demographicsDf.head())

otc_dataInstruments = gc.open_by_url('https://docs.google.com/spreadsheets/d/1H3I-wfVjNQFlUhxkkUF58cWD67xTY16Dmtya5HcA3Wg/edit#gid=1444543832')
otc_dataInstruments = otc_dataInstruments.worksheet('OTCDataInstruments')
otc_dataInstrumentsDf = pd.DataFrame(otc_dataInstruments.get_all_records())
print(otc_dataInstrumentsDf.head())

# Recreate the Summary Table in the PS

def get_summary(data):

  salesTotal = data.groupby('brand')['sales_'].sum()
  data['fullPrice'] = data.price_ + data.prom_
  priceAvg = data.groupby(['brand'])['price_'].mean()
  fullPriceAvg = data.groupby(['brand'])['fullPrice'].mean()
  promoAvg = data.groupby(['brand'])['prom_'].mean()
  costAvg = data.groupby(['brand'])['cost_'].mean()
  marketShare = salesTotal/sum(salesTotal)

  sumTable = pd.concat((salesTotal, marketShare, priceAvg, promoAvg, fullPriceAvg, costAvg), axis=1)
  sumTable = pd.DataFrame(sumTable)
  sumTable.columns = ['salesTotal', 'marketShare', 'priceAvg', 'promoAvg', 'fullPriceAvg', 'costAvg']
  sumTable['sizeTab'] = [25,50,100,25,50,100,25,50,100,50,100]
  sumTable['brandName'] = ['Tylenol', 'Tylenol', 'Tylenol', 'Advil', 'Advil', 'Advil', 'Bayer', 'Bayer', 'Bayer', 'Store', 'Store'] 
  sumTable['brandID'] = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
  
  return(sumTable)

sumTable = get_summary(otc_dataDf)
sumTable.to_latex("summary_statistics.tex")

# make a copy of the sales column
adjusted_sales = otc_dataDf.sales_ 
# and insert it in dataframe   
otc_dataDf.insert(loc=4, column='adjusted_sales', value=adjusted_sales)

def adjust_sales(data):
  '''
  3 package sizes - normalize to 25 tabs for market share calculations
  '''

  # divide 50 tab sales by 2
  tab_50 = [2,5,8,10]
  for i in tab_50:
    data.loc[data['brand'] == i, 'adjusted_sales'] = data['adjusted_sales']/2

  # divide 100 tab sales by 4
  tab_100 = [3,6,9,11]
  for i in tab_100:
    data.loc[data['brand'] == i, 'adjusted_sales'] = data['adjusted_sales']/4

  return(data)

otc_dataDf = adjust_sales(otc_dataDf)

def get_shares(data):

  # calucalte adj. shares by market (store-week)
  data['quant_store_week'] = data.groupby(['store', 'week'])['adjusted_sales'].transform('sum')
  # calculate weekly share
  data['weekly_share'] =  data['adjusted_sales'] / data['quant_store_week']
  # calculate outside share
  data['outshr'] = 1-(data['quant_store_week']/data['count'])
  data['weekly_share'] = data['weekly_share']*(1-data['outshr'])

  # generate logged relative purchase probabilities
  data['Y'] = np.log(data['weekly_share']) - np.log(data['outshr'])

  return data

otc_dataDf = get_shares(data = otc_dataDf)

otc_dataDf.head()

# dummy for branded product
otc_dataDf['branded'] = 1
otc_dataDf.loc[otc_dataDf['brand'] == 10, 'branded'] = 0
otc_dataDf.loc[otc_dataDf['brand'] == 11, 'branded'] = 0

# constant because thats what the example did
otc_dataDf["cons"] = 1

# market variable
otc_dataDf['market_ids'] = otc_dataDf['store'].astype(str) + "." + otc_dataDf['week'].astype(str)
otc_dataDf['market_ids'].astype(str)

# rename some variables
otc_dataDf = otc_dataDf.rename({'price_': 'prices', 'weekly_share': 'shares'}, axis='columns')

# Merge dfs

def combine_data(otc_dataDf, instruments, demographics, sumTable):
 
  # brand name
  otc_dataDf = pd.merge(otc_dataDf, sumTable['brandID'], left_on=['brand'], right_index=True)
  # instruments
  merged = pd.merge(otc_dataDf, instruments, left_on=["brand", "store", "week"], right_on=["brand", "store", "week"])
  # demographics
  full_data = pd.merge(merged, demographics, left_on=["store", "week"], right_on=["store", "week"])

  return(full_data)

full_data = combine_data(otc_dataDf, otc_dataInstrumentsDf, otc_demographicsDf,  sumTable=sumTable)

from google.colab import drive
drive.mount('/content/drive')

# rename some variables
full_data = full_data.rename({'brand': 'product_ids'}, axis='columns')
full_data.head()

"""## 1. Logit"""

#1.1: Estimate using OLS with price and promotion as product characteristics.
res_log1 = smf.ols('Y ~ prices + prom_', data=otc_dataDf).fit()
res_log1.summary().as_latex()
# print(res_log1.summary())

#1.2: Estimate using OLS with price and promotion as product characteristics and brand dummies.
res_log2 = smf.ols('Y ~ prices + prom_ + C(brand)', data=otc_dataDf).fit()
res_log2.summary().as_latex()

#1.3. Using OLS with price and promotion as product characteristics and store-brand 
#(the interaction of brand and store) dummies.

res_log3 = smf.ols('Y ~ prices + prom_ + C(brand)*C(store)', data=otc_dataDf).fit()
res_log3.summary().as_latex()

outputOLS = summary_col([res_log1,res_log2,res_log3],stars=False)
outputOLS

"""4. Estimate the models of 1, 2 and 3 using wholesale cost as an instrument."""

# OLS with price and promotion as product characteristics, using wholesale cost as instrument
wholeSale_IV1 = iv.IV2SLS.from_formula('Y ~ 1 + [prices ~ cost_] + prom_ ', data=otc_dataDf).fit()
wholeSale_IV1.summary.as_latex()
#print(wholeSale_IV1.first_stage)

# OLS with price and promotion as product characteristics and brand dummies
# using wholesale cost as instrument

wholeSale_IV2 = iv.IV2SLS.from_formula('Y ~ 1 + [prices ~ cost_] + prom_ + C(brand)', data=otc_dataDf).fit()
wholeSale_IV2.summary.as_latex()
#print(wholeSale_IV2.first_stage)

# OLS with price and promotion as product characteristics and brand dummies
# using wholesale cost as instrument

wholeSale_IV3 = iv.IV2SLS.from_formula('Y ~ 1 + [prices ~ cost_] + prom_ + C(brand)*C(store)', data=otc_dataDf).fit()
wholeSale_IV3.summary.as_latex()

# blah = pd.read_csv(foo)
#print(wholeSale_IV3.first_stage)

"""5. Estimate the models of 1, 2 and 3 using Hausman instrument."""

hausman_IV1 = iv.IV2SLS.from_formula('Y ~ 1 + [prices ~ pricestore1 + pricestore2 + pricestore3 + pricestore4 + \
pricestore5 + pricestore6 + pricestore7 + pricestore8 + pricestore9 + pricestore10 + pricestore11 + pricestore12 \
+ pricestore13 + pricestore14 + pricestore15 + pricestore16 + pricestore17 + pricestore18 + pricestore19 + \
pricestore20 + pricestore21 + pricestore22 + pricestore23 + pricestore24 + pricestore25 + pricestore26 +\
 pricestore27 + pricestore28 + pricestore29 + pricestore30] + prom_', data=full_data).fit()
hausman_IV1.summary.as_latex()
#print(hausman_IV1.first_stage)

hausman_IV2 = iv.IV2SLS.from_formula('Y ~ 1 + [prices ~ pricestore1 + pricestore2 + pricestore3 + pricestore4 + pricestore5 + pricestore6 + pricestore7 + pricestore8 + pricestore9 + pricestore10 + pricestore11 + pricestore12 + pricestore13 + pricestore14 + pricestore15 + pricestore16 + pricestore17 + pricestore18 + pricestore19 + pricestore20 + pricestore21 + pricestore22 + pricestore23 + pricestore24 + pricestore25 + pricestore26 + pricestore27 + pricestore28 + pricestore29 + pricestore30] + prom_ + C(product_ids) ', data=full_data).fit()
hausman_IV2.summary.as_latex()
#print(hausman_IV2.first_stage)

hausman_IV3 = iv.IV2SLS.from_formula('Y ~ 1 + [prices ~ pricestore1 + pricestore2 + pricestore3 + pricestore4 + pricestore5 + pricestore6 + pricestore7 + pricestore8 + pricestore9 + pricestore10 + pricestore11 + pricestore12 + pricestore13 + pricestore14 + pricestore15 + pricestore16 + pricestore17 + pricestore18 + pricestore19 + pricestore20 + pricestore21 + pricestore22 + pricestore23 + pricestore24 + pricestore25 + pricestore26 + pricestore27 + pricestore28 + pricestore29 + pricestore30] + prom_ + C(product_ids)*C(store)', data=full_data).fit()
hausman_IV3.summary.as_latex()
#print(hausman_IV3.first_stage)

def problem1_6(
    model,
    clean_data=otc_dataDf
               ):
  '''
  Take parameter estimates from given model
  Use the analytic form for logit elasticity to calculate elasticities
  Analytic form for logit elasticity from Levin's notes (slide 13)
  https://web.stanford.edu/~jdlevin/Econ%20257/Demand%20Estimation%20Slides%20B.pdf
  '''
  brand_means = otc_dataDf.groupby(by='brand').mean()
  brand_means
  # otc_dataDf

  model_elasticities = \
  1 * model.params['prices'] \
    * brand_means['prices'] \
    * (1 - brand_means['shares'])

  return(model_elasticities)

# list all logit models
model_list = [res_log1, res_log2, res_log3, 
              wholeSale_IV1, wholeSale_IV2, wholeSale_IV3, 
              hausman_IV1, hausman_IV2, hausman_IV3]

# new dataframe for results
elasticities_df = pd.DataFrame(columns=['OLS1', 'OLS2', 'OLS3', 
                                        'IV4.1', 'IV4.2', 'IV4.3',
                                        'IV5.1', 'IV5.2', 'IV5.3'])

# run elasticity function on all models
columnnr = 0
for i in model_list:
  #print(elasticities_df.columns[columnnr])
  elasticities_df[elasticities_df.columns[columnnr]] = problem1_6(model=i, clean_data=otc_dataDf)
  columnnr += 1

elasticities_df.to_latex()

"""## 2: Random-Coefficients Logit, a.k.a. BLP"""

#Manual BLP:
X1 = np.hstack((otc_dataDf[["prices"]], pd.get_dummies(otc_dataDf["brand"])))

# non-linear, note order is different from Nevo paper
X2 = otc_dataDf[["cons", "prices", "prom_"]].to_numpy()

k = 2

# price
p = otc_dataDf.prices.values

# number of goods per market
J = otc_dataDf.groupby("market_ids").sum().cons.values

# number of simulations per market
N = 3

# number of markets
T = len(J)

# find the share of the outside good
# otc_dataDf["outside"] = .9

# initial delta_0 estimate: log(share) - log(share outside good)
delta_0 = np.log(full_data["shares"]) - np.log(full_data["outshr"])

# markets for itj
markets = otc_dataDf.market_ids.values

# unique markets
marks = np.unique(otc_dataDf.market_ids)

# firms
firms = np.reshape(otc_dataDf.brand.values, (-1,1))

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
V = np.reshape(np.random.standard_normal((k + 1) * N * T), (T * N, k + 1))

# draws for income if same draws in every market
otc_demographicsDf["average_hhincome"] = otc_demographicsDf[[i for i in full_data.columns if i[:-2] == "hhincome"]].mean(axis=1)
incomeMeans = otc_demographicsDf["average_hhincome"].values

demog = incomeMeans
demog2 = demog * demog

demogDf = pd.DataFrame()
demogDf['demog'] = demog
demogDf['demog2'] = demog2

demog = demogDf.to_numpy()

sigma_v = np.std(incomeMeans)
m_t = np.repeat(incomeMeans, N)

#@cuda.jit(nopython = True, parallel = True)
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
                # if N * t + i < demog.shape[0]:
                # calculate utility
                #  log of the numerator of equation (11) from Aviv's RA guide
                out[tj, i] = delta[tj] + \
                x2[tj,0] * (v[N * t + i , 0] * sigma[0] +
                             np.dot(pi[0, :], demog[ t,:])) + \
                x2[tj, 1] * (v[N * t + i, 1] * sigma[1]  +
                            np.dot(pi[1,:], demog[ t ,:])) + \
                x2[tj, 2] * (v[N * t + i, 2] * sigma[2]  +
                            np.dot(pi[2,:], demog[ t ,:])) 
                # else:
                # print(f"{t + i}")

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

#@cuda.jit
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

#@cuda.jit
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

#@cuda.jit
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

# This is the objective function that we optimize the non-linear parameters over
def objective(params, s, x1, x2, v, p, demog, J, T, N, marks, markets, tol, 
              Z, weigh, firms):
    
    # optim flattens the params, so we have to redefine inside 
    sigma = params[0:3]

    alpha = sigma[-1]
    
    pi = params[3:].reshape((3,2))
    
    # number of observation JxT
    obs = np.sum(J)
    
    # force these params to be 0:
    
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

demand_inst_cols = [i for i in otc_dataInstrumentsDf.columns if i[:10] == "pricestore"]
Z = np.hstack((otc_dataInstrumentsDf[demand_inst_cols], pd.get_dummies(full_data["product_ids"])))

# linear parameters
#  this is the FOC on page 5 of Aviv's Appendix
# compare parameters to Table iv of BLP
#  first 5 are the demand side means
#  last 6 are the cost side params


# initial estimates for sigma
sigma = [0, 0, 0]

# initial estimates for pi, by row
pi1 = [0, 0, 0]
pi2 =   [0, 0, 0]


# optim must read all params in as a one dimensional vector
params = np.hstack((sigma, pi1, pi2))

# Recommended initial weighting matrix from Aviv's appendix
w1 = np.linalg.inv(Z.T @ Z)



y2 = d.delta.reshape((-1, 1))
b = np.linalg.inv(X1.T @ Z @ Z.T @ X1) @ (X1.T @ Z @ Z.T @ y2)
b

obj = objective(params,
                full_data.shares.values,
                X1, X2,
                V, p, demog,
                J, T, N,
                marks, markets, 1e-6,
                Z, w1, firms)

pi = params[3:].reshape((3,2))


util = compute_indirect_utility(X2, V, p, demog, d.delta, 
                         sigma, pi, J, T, N)
# 
# 
# 
share = compute_share(X2, V, p, 
                      demog, d.delta, sigma, pi, 
                      J, T, N)
# 
delta = solve_delta(full_data.shares.values,
                    X2, V, p, 
                    demog, d.delta, sigma, pi, 
                    J, T, N, 1e-4)

delta

res_init_wt = minimize(objective,
                      params, 
                      args = (full_data.shares.values, 
                            X1, X2, 
                            V, p, demog, 
                            J, T, N, 
                            marks, markets, 1e-4, 
                            Z, w1, firms), 
                      method = "Nelder-Mead")

res_init_wt

obs = np.sum(J)

# approximate optimal weighting matrix
params_2 = res_init_wt.x
sigma_2 = params_2[0:4]
pi_2 = params_2[3:].reshape((3,2))


# calculate mean utility given the optimal parameters (with id weighting matrix)
d.delta = solve_delta(full_data.shares.values,
                    X2, V, p, 
                    demog, d.delta, sigma_2, pi_2, 
                    J, T, N, 1e-4)

# since we are using both demand and supply side variables,
#  we want to stack the estimated delta and estimated mc
y2 = d.delta.reshape((-1,1))

# this is the first order condition that solves for the linear parameters
b = np.linalg.inv(X1.T @ Z @ w1 @ Z.T @ X1) @ (X1.T @ Z @ w1 @ Z.T @ y2)

# obtain the error
xi_w = y2 - X1 @ b

# update weighting matrix
g_ind = Z * xi_w
vg = g_ind.T @ g_ind / obs

# obtain optimal weighting matrix
weight = np.linalg.inv(vg)

res = minimize(objective,
              params, 
              args = (full_data.shares.values, 
                    X1, X2, 
                    V, p, demog, 
                    J, T, N, 
                    marks, markets, 1e-4, 
                    Z, weight, firms), 
              method = "Nelder-Mead")

params_3

params_3 = res.x
sigma_3 = params_3[0:3]
pi_3 = params_3[3:].reshape((3,2))

# obtain the actual implied quantities and shares from converged delta
q_s = compute_share(X2, V, p, demog, d.delta, sigma_3, pi_3, J, T, N)

delta

q_s[1]

sigma_3

pd.DataFrame(pi_3)

b

full_data_w10 = full_data.loc[full_data['week']==10].reset_index()
full_data_w10_store9 = full_data_w10[full_data_w10['store']==9].reset_index()
full_data_w10_store9['relative_share'] = full_data_w10_store9['shares']/full_data_w10_store9['shares'].sum()
full_data_w10_store9['average_income'] = full_data_w10_store9[[i for i in full_data_w10_store9.columns if i[:2] == "hh"]].values.mean()
full_data_w10_store9

sigma_I = 0.005161
sigma_I_sq = -0.121263
alpha = 1.67671625

def calculate_elasticity_matrix(full_data_w10_store9=full_data_w10_store9):
  my_etas = []
  Q2_elasticty_matrix = pd.DataFrame()
  for j in full_data_w10_store9.index:
    eta_j = []
    for k in full_data_w10_store9.index:
      if j != k:
        eta = (full_data_w10_store9['prices'][k]/full_data_w10_store9['shares'][k]) * \
          (alpha + sigma_I * full_data_w10_store9['average_income'][k] + sigma_I_sq * full_data_w10_store9['average_income'][k]**2) * \
          (full_data_w10_store9['shares'][k] * full_data_w10_store9['shares'][j])
      if j == k:
        eta = (full_data_w10_store9['prices'][k]/full_data_w10_store9['shares'][k]) * \
          (alpha + sigma_I * full_data_w10_store9['average_income'][k] + sigma_I_sq * full_data_w10_store9['average_income'][k]**2) * \
          (full_data_w10_store9['shares'][k] * full_data_w10_store9['shares'][j] + full_data_w10_store9['shares'][k])
      eta_j.append(eta)
    Q2_elasticty_matrix[f'{j}'] = eta_j
  return Q2_elasticty_matrix


Q2_elasticty_matrix = calculate_elasticity_matrix()
Q2_elasticty_matrix

def calculate_mc(df, elasticity_df=Q2_elasticty_matrix):
  elasticities = elasticity_df.values.diagonal()
  shares = df['shares']
  prices = df['prices']
  marginal_costs = []
  k = 0
  while k < len(df.index):
    mc = 1 / elasticities[k] * shares[k] / prices[k] + prices[k]
    marginal_costs.append(mc)
    k += 1
  return marginal_costs

marginal_costs = calculate_mc(df=full_data_w10_store9)
marginal_costs

marginal_costs_df = pd.DataFrame(data=marginal_costs)
marginal_costs_df.to_latex()

mc = 1 / (elasticity[k] * share[k] / price[k])

df = full_data_w10_store9
elasticity_df = Q2_elasticty_matrix
elasticities = elasticity_df.values.diagonal()


marginal_costs

len(elasticities)

# Define the function to retrieve costs
def find_costs(nobs, ahat, price, share, owner):
    """Solve for marginal costs
    nobs = number of products in market
    ahat = estimated price coefficient
    price = vector of prices in market
    share = market shares in market
    owner = ownership vector 
    """
    # Initialize dsdp
    temp = np.zeros(shape=(nobs,nobs))
    dsdp = pd.DataFrame(temp)

    # dsdp matrix where element (row=j,col=r) = ds_r/dp_j
    for j in range(nobs):
        for r in range(nobs):
            if (owner.at[j]==owner.at[r]):
                if (j==r):
                    dsdp[j][r] = ahat*share.at[j]*(1-share.at[j])
                else:
                    dsdp[j][r] = -ahat*share.at[j]*share.at[r]
        
    # Apply inverse. If you've had linear algebra, you'll see what I'm doin.
    # If not, take my word for it: We're just rearranging the system of FOCs
    # to solve for c just like you would in the monopolist case.
    inv_dsdp = np.linalg.inv(dsdp) 
      
    # Solve for dollar markups.
    markup = -np.dot(inv_dsdp,share)
    
    # Solve for chat 
    chat = price - markup 
    
    return chat

parans_M1 = res_log3.params
parans_M1

# restrict data to week 10
full_data_w10 = full_data.loc[full_data['week']==10].reset_index()
full_data_w10['chat'] = 0

# Identify inputs
ahat = parans_M1['prices']
price = full_data_w10['prices']
share = full_data_w10['shares']
owner = full_data_w10['brandID']
nobs = len(share)

full_data_w10['chat'] = find_costs(nobs, ahat, price, share, owner)
full_data_w10['chat'].describe()

# Define function to solve for market shares
def mkt_shar(nobs,ahat,price,util):
    numerator = np.exp(ahat*price + util)
    denominator = sum(numerator)
    denominator = denominator + 1
    share = numerator/denominator
    return share

# Define the system of FOCs
def FOC(price, nobs, ahat, util, owner, chat):
    """Solve for the FOCs at a price guess
    nobs = number of products in market
    ahat = estimated price coefficient
    price = vector of prices in market
    util = X-times-Beta
    owner = ownership vector 
    chat = marginal cost estimate
    """
    
    # Solve for market share conditional on price
    share = mkt_shar(nobs,ahat,price,util)
    
    # Initialize dsdp
    temp = np.zeros(shape=(nobs,nobs))
    dsdp = pd.DataFrame(temp)

    # dsdp matrix where element (row=j,col=r) = ds_r/dp_j
    for j in range(nobs):
        for r in range(nobs):
            if (owner.at[j]==owner.at[r]):
                if (j==r):
                    dsdp[j][r] = ahat*share.at[j]*(1-share.at[j])
                else:
                    dsdp[j][r] = -ahat*share.at[j]*share.at[r]
        
    # Apply inverse. If you've had linear algebra, you'll see what I'm doin.
    # If not, take my word for it: We're just rearranging the system of FOCs
    # to solve for c just like you would in the monopolist case.
    inv_dsdp = np.linalg.inv(dsdp) 
      
    # Solve for dollar markups.
    markup = -np.dot(inv_dsdp,share)
    
    # Solve for residual. If the FOCs jointly hold, this is a vector of zeros
    resid = price - (chat + markup)
    
#    print(sum(resid))
          
    return resid

# Solve for new prices using a numerical equation solver (fsolve)
import time
from scipy.optimize import fsolve

# Account for the merger via the ownership matrix
full_data_w10.loc[(full_data_w10['brandID']== 2),'brandID'] = 1
full_data_w10.loc[(full_data_w10['brandID']== 3),'brandID'] = 1

# Define util (X-times-Beta)
util = full_data_w10['Y'] - ahat*price

chat = full_data_w10['chat']
p0 = price # initial guess of equilibrium prices (prices from data)

"""This is a faster way to solve

"""

def contraction(price, nobs, ahat, util, owner, chat):
    """Solve for the FOCs at a price guess
    nobs = number of products in market
    ahat = estimated price coefficient
    price = vector of prices in market
    util = X-times-Beta
    owner = ownership vector 
    chat = marginal cost estimate
    """
    iter = 1
    maxiter = 1000
    norm = 1
    tol = 1e-6
    
    pnew = price # Initial guess
    
    while (iter<=maxiter) & (norm > tol):
        
        pold = pnew
        
        # Solve for market share conditional on price
        share = mkt_shar(nobs,ahat,pold,util)

        # Initialize dsdp
        temp = np.zeros(shape=(nobs,nobs))
        dsdp = pd.DataFrame(temp)

        # dsdp matrix where element (row=j,col=r) = ds_r/dp_j
        for j in range(nobs):
            for r in range(nobs):
                if (owner.at[j]==owner.at[r]):
                    if (j==r):
                        dsdp[j][r] = ahat*share.at[j]*(1-share.at[j])
                    else:
                        dsdp[j][r] = -ahat*share.at[j]*share.at[r]

        # Apply inverse. If you've had linear algebra, you'll see what I'm doin.
        # If not, take my word for it: We're just rearranging the system of FOCs
        # to solve for c just like you would in the monopolist case.
        inv_dsdp = np.linalg.inv(dsdp) 

        # Solve for dollar markups.
        markup = -np.dot(inv_dsdp,share)

        # Solve for residual. If the FOCs jointly hold, this is a vector of zeros
        pnew = chat + markup
    
        # Check if we're done
        norm = max(abs(pnew-pold))
        iter = iter + 1
        #print(norm)
          
    return pnew

# Solve the faster way
start_time = time.time()  # Start timer
p_prime2 = contraction(price, nobs, ahat, util, owner, chat)
finish_time = time.time()  # end timer
print('FOC Algorithm Finished. Execution Time: {0:.2f} seconds'.format(finish_time-start_time))

pchange = 100*(p_prime2/price - 1)  # Percentage change
p_delta = p_prime2-price

# Raw stats
pchange.describe()

# Need to merge prices back into frame

# price changes in week 10 - store 9 
p_prime2.name = 'new_prices'
df = pd.concat([full_data_w10, p_prime2], axis=1)
df = pd.concat([df, p_delta], axis=1)
df = df.rename({0: 'price_change', "product_ids" : "brand"}, axis='columns')
df[df["store"] == 9][['store', 'week', 'brand', 'prices', 'new_prices', 'price_change']]
