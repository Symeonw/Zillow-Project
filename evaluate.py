import stats
import stats_functions
df = data("tips")
df.info()
total_bill = tips["total_bill"]
tips = tips["tips"]
regr = old('tip ~ total_bill', data=tips).fit()
tips["yhat"] = regr.predict(tips["total_bill"])
def plot_residuals(x,y,df):
    sns.residplot(x,y,data=df)
    plt.show()

def regression_errors(y, y_hat):
    sse = ((y-y_hat)**2).sum()
    ess = ((yhat-y.mean())**2).sum()
    tss = sse + ess
    mse = mean_squared_error(y, y_hat)
    rmse = mse ** 0.5
    return sse, ess, tss, mse, rmse

def baseline_mean_errors(y, y_hat):
    sse = ((y-y_hat)**2).sum()
    mse = mean_squared_error(y, y_hat)
    rmse = mse ** 0.5
    return sse, mse, rmse

def better_than_baseline(sse, sse_baseline):
    return sse > sse_baseline

def model_significance(ols_model):
    return ols_model.rsquared, ols_model.f_pvalue
    

