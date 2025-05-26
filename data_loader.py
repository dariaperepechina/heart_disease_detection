import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the dataset files
        """
        self.data_dir = data_dir
        self.main_data_path = os.path.join(data_dir, 'dat.csv')
        self.metadata_path = os.path.join(data_dir, 'dat_md.csv')
        self.dictionary_path = os.path.join(data_dir, 'dataDictionary.csv')
        
    def load_data(self):
        """
        Load and merge the dataset files.
        
        Returns:
            Merged DataFrame
        """
        print(f"Loading data from {self.data_dir}...")
        
        main_data = pd.read_csv(self.main_data_path)
        print(f"Main data loaded: {main_data.shape[0]} rows, {main_data.shape[1]} columns")
        
        if os.path.exists(self.metadata_path):
            metadata = pd.read_csv(self.metadata_path)
            print(f"Metadata loaded: {metadata.shape[0]} rows, {metadata.shape[1]} columns")
            
            common_cols = set(main_data.columns) & set(metadata.columns)
            if common_cols:
                merge_col = list(common_cols)[0]
                print(f"Merging on column: {merge_col}")
                merged_data = pd.merge(main_data, metadata, on=merge_col, how='left')
            else:
                print("No common columns found for merging. Using main data only.")
                merged_data = main_data
        else:
            print("Metadata file not found. Using main data only.")
            merged_data = main_data
        
        if os.path.exists(self.dictionary_path):
            data_dict = pd.read_csv(self.dictionary_path)
            print(f"Data dictionary loaded: {data_dict.shape[0]} rows, {data_dict.shape[1]} columns")
            
            print("\nColumn Descriptions:")
            for _, row in data_dict.iterrows():
                if 'Variable' in row and 'Description' in row:
                    print(f"  {row['Variable']}: {row['Description']}")
        
        return merged_data
    
    def identify_column_types(self, df):
        """
        Identify column types based on data content.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with column types
        """
        numerical_columns = []
        categorical_columns = []
        text_columns = []
        
        for col in df.columns:
            if 'id' in col.lower() or 'identifier' in col.lower():
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() < min(20, len(df) * 0.05):
                    categorical_columns.append(col)
                else:
                    numerical_columns.append(col)
            else:
                if df[col].dtype == 'object':
                    if (df[col].nunique() > 20 or 
                        df[col].str.len().mean() > 50):
                        text_columns.append(col)
                    else:
                        categorical_columns.append(col)
        
        target_candidates = [col for col in df.columns if col.lower() in 
                            ['diagnosis', 'target', 'label', 'heart_disease', 
                             'disease', 'condition', 'outcome']]
        
        target_column = target_candidates[0] if target_candidates else None
        
        return {
            'numerical_columns': numerical_columns,
            'categorical_columns': categorical_columns,
            'text_columns': text_columns,
            'target_column': target_column
        }

if __name__ == "__main__":
    loader = DataLoader('data')
    df = loader.load_data()
    column_types = loader.identify_column_types(df)
    
    print("\nIdentified Column Types:")
    for type_name, columns in column_types.items():
        print(f"\n{type_name}:")
        for col in columns:
            print(f"  - {col}")
