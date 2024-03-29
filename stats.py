import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,explained_variance_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression, LassoCV
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
import matplotlib
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from pydataset import data
from statsmodels.formula.api import ols
import re
import multiprocessing as mp
