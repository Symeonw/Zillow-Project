import stats
import stats_functions
from scipy import stats
df, info, head = prepare_df()
train, test, X_train, y_train, X_test, y_test, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = split_data(df)
#Null: no corr exists between bathroom_count and bedroom_count
#alt: corr exists between bathroom_count and bedroom_count
means = df.bathroom_count.mean()
xbar = df.bedroom_count.mean()
s = df.bedroom_count.std()
n = df.bedroom_count.shape[0]
degf = n - 1
standard_error = s / sqrt(n)
t = (xbar - means)/(s / sqrt(n))
p = stats.t(degf).sf(t) * 2
p,t

#show corr between variables
def corr_chart():
    corr = df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap="bone_r", vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(df.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.columns)
    return plt.show

#one of the strongest corr is bettwen bathroom_count and square_feet
#all variables appear to be correlated to the dependent, though fireplace_count, bathroom_count and bedroom_count are the 3 highest (when compaired independently)
#bathroom_count and home_value have a corr of .53, bathroom_count and square_feet have a corr of .85, and bedroom_count and square_feet share at .54 the 3 top highest corr's. 
#summary - bathroom_count, bedroom_count, and fireplace_count should be best predictors of home_value, though more research would be required to support this hypothesis. 
