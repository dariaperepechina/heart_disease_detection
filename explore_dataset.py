import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

class DataExplorer:
    def __init__(self, output_dir='data_exploration'):
        """
        Initialize the data explorer.
        
        Args:
            output_dir: Directory to save exploration results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def explore(self, data_path):
        """
        Explore the dataset and generate visualizations.
        
        Args:
            data_path: Path to the data file
        """
        print(f"Exploring dataset: {data_path}")
        
        df = self._load_data(data_path)
        
        self._save_basic_info(df)
        
        self._visualize_distributions(df)
        self._visualize_correlations(df)
        self._visualize_missing_values(df)
        
        self._identify_potential_targets(df)
        
        self._analyze_feature_importance(df)
        
        self._visualize_pca(df)
        
        print(f"Exploration completed. Results saved to {self.output_dir}")
    
    def _load_data(self, data_path):
        """
        Load data from file.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        file_ext = os.path.splitext(data_path)[1].lower()
        
        encodings = ['utf-8', 'latin1', 'cp1251', 'ISO-8859-1']
        
        if file_ext == '.csv':
            for encoding in encodings:
                try:
                    df = pd.read_csv(data_path, encoding=encoding)
                    print(f"Successfully loaded CSV with encoding: {encoding}")
                    return df
                except UnicodeDecodeError:
                    continue
            raise UnicodeDecodeError(f"Failed to decode file with any of the encodings: {encodings}")
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(data_path)
        elif file_ext == '.json':
            for encoding in encodings:
                try:
                    df = pd.read_json(data_path, encoding=encoding)
                    print(f"Successfully loaded JSON with encoding: {encoding}")
                    return df
                except UnicodeDecodeError:
                    continue
            raise UnicodeDecodeError(f"Failed to decode file with any of the encodings: {encodings}")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return df
    
    def _save_basic_info(self, df):
        """
        Save basic information about the dataset.
        
        Args:
            df: Input DataFrame
        """
        with open(os.path.join(self.output_dir, 'basic_info.txt'), 'w') as f:
            f.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
            
            f.write("Data Types:\n")
            f.write(df.dtypes.to_string())
            f.write("\n\n")
            
            f.write("Summary Statistics:\n")
            f.write(df.describe().to_string())
            f.write("\n\n")
            
            f.write("Missing Values:\n")
            missing = df.isnull().sum()
            f.write(missing[missing > 0].to_string())
            f.write("\n\n")
            
            f.write("Sample Data:\n")
            f.write(df.head().to_string())
        
        df.describe().to_csv(os.path.join(self.output_dir, 'summary_statistics.csv'))
        
        column_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Min': [df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else None for col in df.columns],
            'Max': [df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None for col in df.columns],
            'Mean': [df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None for col in df.columns],
            'Std': [df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else None for col in df.columns]
        })
        column_info.to_csv(os.path.join(self.output_dir, 'column_info.csv'), index=False)
    
    def _visualize_distributions(self, df):
        """
        Visualize distributions of numerical and categorical variables.
        
        Args:
            df: Input DataFrame
        """
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_cols) > 0:
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(numerical_cols):
                if i >= 15: 
                    break
                    
                plt.subplot(3, min(5, max(1, len(numerical_cols))), i + 1)
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f'Distribution of {col}')
                plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, 'numerical_distributions.png'))
            plt.close()
            
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(numerical_cols):
                if i >= 15:  
                    break
                    
                plt.subplot(3, min(5, max(1, len(numerical_cols))), i + 1)
                sns.boxplot(y=df[col].dropna())
                plt.title(f'Boxplot of {col}')
                plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, 'numerical_boxplots.png'))
            plt.close()
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(categorical_cols):
                if i >= 15:  
                    break
                    
                plt.subplot(3, min(5, max(1, len(categorical_cols))), i + 1)
                value_counts = df[col].value_counts()
                if len(value_counts) > 10:
                    value_counts = value_counts.head(10)
                    plt.title(f'Top 10 values of {col}')
                else:
                    plt.title(f'Distribution of {col}')
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.xticks(rotation=45)
                plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, 'categorical_distributions.png'))
            plt.close()
    
    def _visualize_correlations(self, df):
        """
        Visualize correlations between numerical variables.
        
        Args:
            df: Input DataFrame
        """
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_cols) > 1:
            try:
                corr = df[numerical_cols].corr()
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
                plt.title('Correlation Matrix')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'))
                plt.close()
                
                corr.to_csv(os.path.join(self.output_dir, 'correlation_matrix.csv'))
                
                high_corr_pairs = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        if abs(corr.iloc[i, j]) > 0.7:
                            high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
                
                if high_corr_pairs:
                    plt.figure(figsize=(15, 10))
                    for i, (col1, col2, corr_val) in enumerate(high_corr_pairs[:9]):  
                        plt.subplot(3, 3, i + 1)
                        plt.scatter(df[col1], df[col2], alpha=0.5)
                        plt.title(f'{col1} vs {col2}\nCorr: {corr_val:.2f}')
                        plt.xlabel(col1)
                        plt.ylabel(col2)
                        plt.tight_layout()
                    
                    plt.savefig(os.path.join(self.output_dir, 'high_correlation_pairs.png'))
                    plt.close()
            except Exception as e:
                print(f"Error in correlation visualization: {str(e)}")
    
    def _visualize_missing_values(self, df):
        """
        Visualize missing values.
        
        Args:
            df: Input DataFrame
        """
        missing = df.isnull().sum()
        
        if missing.sum() > 0:
            try:
                plt.figure(figsize=(12, 6))
                missing = missing[missing > 0].sort_values(ascending=False)
                sns.barplot(x=missing.index, y=missing.values)
                plt.title('Missing Values by Column')
                plt.xticks(rotation=45)
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'missing_values.png'))
                plt.close()
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
                plt.title('Missing Value Patterns')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'missing_patterns.png'))
                plt.close()
            except Exception as e:
                print(f"Error in missing values visualization: {str(e)}")
    
    def _identify_potential_targets(self, df):
        """
        Identify potential target variables for modeling.
        
        Args:
            df: Input DataFrame
        """
        potential_targets = []
        
        for col in df.columns:
            n_unique = df[col].nunique()
            
            if n_unique == 2:
                potential_targets.append((col, 'binary', n_unique))
            elif 2 < n_unique <= 10 and df[col].dtype in ['int64', 'float64', 'object', 'category']:
                potential_targets.append((col, 'multi-class', n_unique))
        
        with open(os.path.join(self.output_dir, 'potential_targets.txt'), 'w') as f:
            f.write("Potential Target Variables:\n")
            
            if potential_targets:
                for col, type_, n_unique in potential_targets:
                    f.write(f"{col}: {type_} variable with {n_unique} unique values\n")
                    
                    f.write(f"Distribution:\n")
                    f.write(df[col].value_counts().to_string())
                    f.write("\n\n")
            else:
                f.write("No obvious binary or multi-class variables found.\n")
                f.write("Consider using a numerical variable as a regression target or creating a binary target.\n")
        
        if potential_targets:
            try:
                plt.figure(figsize=(15, 10))
                for i, (col, _, _) in enumerate(potential_targets[:6]): 
                    plt.subplot(2, 3, i + 1)
                    df[col].value_counts().plot(kind='bar')
                    plt.title(f'Distribution of {col}')
                    plt.tight_layout()
                
                plt.savefig(os.path.join(self.output_dir, 'potential_targets.png'))
                plt.close()
            except Exception as e:
                print(f"Error in potential targets visualization: {str(e)}")
    
    def _analyze_feature_importance(self, df):
        """
        Analyze feature importance using mutual information.
        
        Args:
            df: Input DataFrame
        """
        potential_targets = []
        
        for col in df.columns:
            n_unique = df[col].nunique()
            
            if n_unique == 2 or (2 < n_unique <= 10 and df[col].dtype in ['int64', 'float64']):
                potential_targets.append(col)
        
        if not potential_targets:
            print("No suitable target variables found for feature importance analysis.")
            return
        
        target_col = potential_targets[0]
        
        features = df.drop(columns=[target_col])
        
        numerical_features = features.select_dtypes(include=['int64', 'float64'])
        
        if numerical_features.empty:
            print("No numerical features found for feature importance analysis.")
            return
        
        if numerical_features.isnull().values.any() or df[target_col].isnull().values.any():
            print("Data contains missing values. Dropping rows with missing values for feature importance analysis.")
            
            analysis_df = pd.concat([numerical_features, df[target_col]], axis=1)
            
            analysis_df = analysis_df.dropna()
            
            if len(analysis_df) == 0:
                print("After dropping missing values, no data remains for analysis.")
                return
            
            numerical_features = analysis_df.drop(columns=[target_col])
            target = analysis_df[target_col]
        else:
            target = df[target_col]
        
        try:
            if df[target_col].nunique() <= 10: 
                mi_scores = mutual_info_classif(numerical_features, target)
            else:  
                mi_scores = mutual_info_regression(numerical_features, target)
            
            mi_df = pd.DataFrame({
                'Feature': numerical_features.columns,
                'Mutual Information': mi_scores
            }).sort_values('Mutual Information', ascending=False)
            
            mi_df.to_csv(os.path.join(self.output_dir, 'mutual_information.csv'), index=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Mutual Information', y='Feature', data=mi_df)
            plt.title(f'Feature Importance (Mutual Information) for {target_col}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
            plt.close()
            
            print(f"Feature importance analysis completed for target: {target_col}")
        except Exception as e:
            print(f"Error in feature importance analysis: {str(e)}")
    
    def _visualize_pca(self, df):
        """
        Visualize data using PCA.
        
        Args:
            df: Input DataFrame
        """
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_cols) < 3:
            print("Not enough numerical columns for PCA visualization.")
            return 
        
        potential_targets = []
        
        for col in df.columns:
            n_unique = df[col].nunique()
            
            if n_unique == 2 or (2 < n_unique <= 10 and df[col].dtype in ['int64', 'float64']):
                potential_targets.append(col)
        
        df_pca = df.copy()
        
        if df_pca[numerical_cols].isnull().values.any():
            print("Data contains missing values. Dropping rows with missing values for PCA visualization.")
            df_pca = df_pca.dropna(subset=numerical_cols)
            
            if len(df_pca) == 0:
                print("After dropping missing values, no data remains for PCA visualization.")
                return
        
        if potential_targets:
            target_col = potential_targets[0]
            
            if df_pca[target_col].isnull().values.any():
                df_pca = df_pca.dropna(subset=[target_col])
                
                if len(df_pca) == 0:
                    print("After dropping missing values in target, no data remains for PCA visualization.")
                    return
            
            features = df_pca.drop(columns=[target_col])
            target = df_pca[target_col]
        else:
            features = df_pca[numerical_cols]
            target = None
        
        numerical_features = features.select_dtypes(include=['int64', 'float64'])
        
        if numerical_features.empty:
            print("No numerical features remain for PCA visualization.")
            return
        
        try:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numerical_features)
            
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(scaled_data)
            
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            
            plt.figure(figsize=(10, 8))
            
            if target is not None:
                pca_df['target'] = target.values
                
                unique_targets = pca_df['target'].unique()
                
                if len(unique_targets) <= 10: 
                    for target_value in unique_targets:
                        subset = pca_df[pca_df['target'] == target_value]
                        plt.scatter(subset['PC1'], subset['PC2'], label=f'Class {target_value}', alpha=0.7)
                    
                    plt.legend()
                else:
                    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
            else:
                plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
            
            plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('PCA of Dataset')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'pca_visualization.png'))
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, 3), pca.explained_variance_ratio_)
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('Explained Variance by Principal Components')
            plt.xticks([1, 2])
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'pca_explained_variance.png'))
            plt.close()
            
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=['PC1', 'PC2'],
                index=numerical_features.columns
            )
            loadings.to_csv(os.path.join(self.output_dir, 'pca_loadings.csv'))
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('PCA Feature Loadings')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'pca_loadings.png'))
            plt.close()
            
            print("PCA visualization completed successfully.")
        except Exception as e:
            print(f"Error in PCA visualization: {str(e)}")

