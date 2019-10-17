from pydataset import data
import seaborn as sns
import pandas as pd
from statsmodels.formula.api import ols
import wrangle
import matplotlib.pyplot as plt
df = data("tips")
def plot_variable_pairs(df):
    x = sns.PairGrid(df)
    x.map_diag(plt.hist)
    x.map_offdiag(sns.regplot)
    plt.show()
    
def plot_categorical_and_continous_vars(cat_var, cont_var,df):
    barplot = sns.barplot( x = cat_var, y = cont_var, data=df)
    regplot = sns.regplot (x = cat_var, y = cont_var, data=df)
    scatterplot = sns.scatterplot(x = cat_var, y = cont_var, data=df)
    plt.show()


def months_to_years(tenure_months, df):
    df = df.assign(tenure_years = lambda x: x[tenure_months]%12)
    return df