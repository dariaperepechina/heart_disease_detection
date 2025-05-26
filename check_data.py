import pandas as pd
import numpy as np

def check_data(data_path):
    """
    Check the data and print information about it.
    
    Args:
        data_path: Path to the data file
    """
    print(f"Checking data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    print("\nDataFrame info:")
    print(df.info())
    
    print("\nColumn names:")
    print(df.columns.tolist())
    
    print("\nData types:")
    print(df.dtypes)
    
    target_col = df.columns[-1]
    print(f"\nTarget column: {target_col}")
    print(f"Target data type: {df[target_col].dtype}")
    print(f"Target unique values: {df[target_col].unique()}")
    print(f"Target value counts:\n{df[target_col].value_counts()}")
    
    if pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Target is numeric with range: {df[target_col].min()} to {df[target_col].max()}")
    else:
        print("Target is not numeric")
    
    return df

if __name__ == "__main__":
    check_data("data/split/train.csv")
