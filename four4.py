def get_4(file_):
    import pandas as pd
    original_df = pd.read_csv(file_)
    return original_df.head(5), (original_df.shape), original_df.info(), original_df.describe()