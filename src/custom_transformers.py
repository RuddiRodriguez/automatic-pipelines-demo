from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector
import pandas as pd
import os
import pyreadstat
import numpy as np 
from sklearn.model_selection import train_test_split

class LoadData(BaseEstimator, TransformerMixin):
    def __init__(self, path):
        self.path = path
        

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        extension =  os.path.splitext (self.path)[1].lower()
        if extension == '.sav':
            df, meta = pyreadstat.read_sav(self.path)
            return df
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
       
        




class FilterByValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, value, operator='>='):
        self.column = column
        self.value = value
        self.operator = operator

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.operator == '>=':
            mask = X[self.column] >= self.value
        elif self.operator == '<=':
            mask = X[self.column] <= self.value
        elif self.operator == '==':
            mask = X[self.column] == self.value
        elif self.operator == '!=':
            mask = X[self.column] != self.value
        else:
            raise ValueError(f"Invalid operator: {self.operator}")

        if y is not None:
            return X[mask], y[mask]
        else:
            return X[mask]

        

class DropNaNValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column=None):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.column:
            mask = X[self.column].notna()
            return self._apply_mask(X, y, mask)
        else:
            mask = X.notna().all(axis=1)
            return self._apply_mask(X, y, mask)
    def _apply_mask(self, X, y, mask):
        if y is not None:
            return X[mask], y[mask]
        else:
            return X[mask]    
        
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method, include=None, exclude=None, query=None):
        self.method = method
        self.include = include
        self.exclude = exclude
        self.query = query

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.method == 'dtype':
            return X.select_dtypes(include=self.include, exclude=self.exclude)
        
        elif self.method == 'substring':
            selected_cols = [col for col in X.columns if any(sub in col for sub in self.query)]
            return X[selected_cols]

        elif self.method == 'exact':
            return X[self.query]
        
        else:
            raise ValueError(f"Invalid method: {self.method}")   
        
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.columns is None:
            return X
        else:
            return X.drop(columns=self.columns)
            

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method, include=None, exclude=None, query=None):
        self.method = method
        self.include = include
        self.exclude = exclude
        self.query = query

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.method == 'dtype':
            return X.select_dtypes(include=self.include, exclude=self.exclude)
        
        elif self.method == 'substring':
            selected_cols = [col for col in X.columns if any(sub in col for sub in self.query)]
            return X[selected_cols]

        elif self.method == 'exact':
            return X[self.query]
        
        else:
            raise ValueError(f"Invalid method: {self.method}")

        
        

# Custom Pipeline
class CustomPipeline:
    def __init__(self, X_transformers):
        self.X_transformers = X_transformers

    def fit(self, X, y):
        for t in self.X_transformers:
            t.fit(X)
        return self

    def transform(self, X, y):
        X_new, y_new = X, y
        for t in self.X_transformers:
            X_new, y_new = t.transform(X_new, y_new)
        return X_new, y_new

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)
    
    
    
import os
import pyreadstat

class RawDataTransformations (BaseEstimator, TransformerMixin):	
    def __init__(self, path):
        self.path = path
    
    def load_data(self, df=None, **kwargs):
        extension = os.path.splitext(self.path)[1].lower()
        if extension == '.sav':
            df, meta = pyreadstat.read_sav(self.path)
        elif extension == '.csv':
            df = pd.read_csv(self.path)
        else:
            raise ValueError(f'Unsupported file extension: {extension}')
        return df
    
    def dropna_on_target(self, df, columns):
        return df.dropna(subset=[columns])
    
    
    def filter_df (self, df, column, value, operator='>='):
        if operator == '>=':
            mask = df[column] >= value
        elif operator == '<=':
            mask = df[column] <= value
        elif operator == '==':
            mask = df[column] == value
        elif operator == '!=':
            mask = df[column] != value
        else:
            raise ValueError(f"Invalid operator: {operator}")
        return df[mask]
    
    
    def target_binarization_by_percentile (self, df, column, percentile, operator='>='):
        if operator == '>=':
            df[column] = np.where(df[column] >= np.percentile(df[column], percentile), 1, 0)
        elif operator == '<=':
            df[column] = np.where(df[column] <= np.percentile(df[column], percentile), 1, 0)
        elif operator == '==':
            df[column] = np.where(df[column] == np.percentile(df[column], percentile), 1, 0)
        elif operator == '!=':
            df[column] = np.where(df[column] != np.percentile(df[column], percentile), 1, 0)
        else:
            raise ValueError(f"Invalid operator: {operator}")
        return df
    
    def split_train_test(self, df, target):
        X, y = df.drop(target, axis=1), df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test 
    
    
    def feature_selector(self,  df, target , method = 'dtype', include=None, exclude=None, query=None):
        if method == 'dtype':
            return pd.concat([df.select_dtypes(include=include, exclude=exclude), df[target]],axis=1)
        
        elif method == 'substring':
            selected_cols = [col for col in df.columns if any(sub in col for sub in query)]
        
            return pd.concat([df[selected_cols],df[target]],axis=1)

        elif method == 'exact':
            return pd.concat([df[query], df[target]], axis=1)
        
        else:
            raise ValueError(f"Invalid method: {method}")
    
    

        
        
    
    
    # Add other transformation methods    

