import stats_functions
import stats
def prepare_df():
    df = get_df()
    df = df.dropna()
    df.columns = ("bathroom_count", "bedroom_count", "fireplace_count", "year_built", "home_value", "square_feet")
    info = df.info()
    head = df.head()
    df[np.abs(df - df.mean()) <= (3*df.std())]
    return df, info, head

prepare_df()