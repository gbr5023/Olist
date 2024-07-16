# -*- coding: utf-8 -*-
"""
@author: Giselle
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


"""
-----------------------------------
------- 1. CUSTOM FUNCTIONS -------
-----------------------------------
"""

"""
-----------------------------------
----- 1. CUSTOM TRANSFORMERS ------
           1.1 Classes
-----------------------------------
"""

class ColumnMapping(BaseEstimator, TransformerMixin):
    """
    This class applies the map() function into a DataFrame for transforming a columns given a mapping dictionary

    Parameters
    ----------
    :param old_col_name: name of the columns where mapping will be applied [type: string]
    :param mapping_dict: python dictionary with key/value mapping [type: dict]
    :param new_col_name: name of the new column resulted by mapping [type: string, default: 'target]
    :param drop: flag that guides the dropping of the old_target_name column [type: bool, default: True]

    Returns
    -------
    :return X: pandas DataFrame object after mapping application [type: pd.DataFrame]

    Application
    -----------
    # Transforming a DataFrame column given a mapping dictionary
    mapper = ColumnMapping(old_col_name='col_1', mapping_dict=dictionary, new_col_name='col_2', drop=True)
    df_mapped = mapper.fit_transform(df)
    """

    def __init__(self, old_col_name, mapping_dict, new_col_name='target', drop=True):
        self.old_col_name = old_col_name
        self.mapping_dict = mapping_dict
        self.new_col_name = new_col_name
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Applying mapping
        X[self.new_col_name] = X[self.old_col_name].map(self.mapping_dict)

        # Dropping the old columns (if applicable)
        if self.drop:
            X.drop(self.old_col_name, axis=1, inplace=True)

        return X


def import_data(path, sep=',', optimized=True, n_lines=50, encoding='utf-8', usecols=None, verbose=True):
    """
    This functions applies a csv reading in an optimized way, converting data types (float64 to float32 and
    int 64 to int32), reducing script memory usage.

    Parameters
    ----------
    :param path: path reference for importing the data [type: string]
    :param sep: separator parameter for read_csv() method [type: string, default: ',']
    :param optimized: boolean flag for reading data in an optimized way [type: bool, default: True]
    :param n_lines: number of lines read during the data type optimization [type: int, default: 50]
    :param encoding: encoding param for read_csv() method [type: string, default: 'utf-8']
    :param verbose: the verbose arg allow communication between steps [type: bool, default: True]
    :param usecols: columns to read - set None to read all the columns [type: list, default: None]

    Return
    ------
    :return: df: file after the preparation steps [type: pd.DataFrame]

    Application
    -----------
    # Reading the data and applying a data type conversion for optimizing the memory usage
    df = import_data(filepath, optimized=True, n_lines=100)
    """

    # Validating the optimized flag for optimizing memory usage
    if optimized:
        # Reading only the first rows of the data
        df_raw = pd.read_csv(path, sep=sep, nrows=n_lines, encoding=encoding, usecols=usecols)
        start_mem = df_raw.memory_usage().sum() / 1024 ** 2

        # Columns were the optimization is applicable
        float64_cols = [col for col, dtype in df_raw.dtypes.items() if dtype == 'float64']
        int64_cols = [col for col, dtype in df_raw.dtypes.items() if dtype == 'int64']
        total_opt = len(float64_cols) + len(int64_cols)
        if verbose:
            print(f'This dataset has {df_raw.shape[1]} columns, which {total_opt} is/are applicable to optimization.\n')

        # Optimizing data types: float64 to float32
        for col in float64_cols:
            df_raw[col] = df_raw[col].astype('float32')

        # Optimizing data types: int64 to int32
        for col in int64_cols:
            df_raw[col] = df_raw[col].astype('int32')

        # Looking at memory reduction
        if verbose:
            print('----------------------------------------------------')
            print(f'Memory usage ({n_lines} lines): {start_mem:.4f} MB')
            end_mem = df_raw.memory_usage().sum() / 1024 ** 2
            print(f'Memory usage after optimization ({n_lines} lines): {end_mem:.4f} MB')
            print('----------------------------------------------------')
            mem_reduction = 100 * (1 - (end_mem / start_mem))
            print(f'\nReduction of {mem_reduction:.2f}% on memory usage\n')

        # Creating an object with new dtypes
        dtypes = df_raw.dtypes
        col_names = dtypes.index
        types = [dtype.name for dtype in dtypes.values]
        column_types = dict(zip(col_names, types))

        # Trying to read the dataset with new types
        try:
            return pd.read_csv(path, sep=sep, dtype=column_types, encoding=encoding, usecols=usecols)
        except ValueError as e1:
            # Error cach during data reading with new data types
            print(f'ValueError on data reading: {e1}')
            print('The dataset will be read without optimization types.')
            return pd.read_csv(path, sep=sep, encoding=encoding, usecols=usecols)
    else:
        # Reading the data without optimization
        return pd.read_csv(path, sep=sep, encoding=encoding, usecols=usecols)


def split_cat_num_data(df):
    """
    This functions receives a DataFrame object and extracts numerical and categorical features from it

    Parameters
    ----------
    :param df: DataFrame object where feature split would be extracted [type: pd.DataFrame]

    Return
    ------
    :return: num_attribs, cat_attribs: lists with numerical and categorical features [type: list]

    Application
    -----------
    # Extracting numerical and categorical features for a given DataFrame
    num_cols, cat_cols = split_cat_num_data(df)
    """

    # Splitting data attributes by data type
    num_attribs = [col for col, dtype in df.dtypes.items() if dtype != 'object']
    cat_attribs = [col for col, dtype in df.dtypes.items() if dtype == 'object']

    return num_attribs, cat_attribs


def calc_working_days(date_series1, date_series2, convert=True):
    """
    This functions receives two date series as args and calculates the working days between each of its rows.

    Parameters
    ----------
    :param date_series1: first date series to be used on working days calculation [type: pd.Series]
    :param date_series2: second date series to subtract the first one [type: pd.Series]
    :param convert: flag that guides the series conversions in datetime objects [type: bool, default: True]

    Return
    ------
    :return: wd_list: list with working days calculations between two date series

    Application
    -----------
    # Calculating the working days between two date series
    working_days = calc_working_days(df['purchase_date'], df['delivered_date'], convert=True)
    """

    # Auxiliar function for threating exceptions during the np.busday_count() function
    def handle_working_day_calc(d1, d2):
        try:
            date_diff = np.busday_count(d1, d2)
            return date_diff
        except:
            return np.NaN

    # Applying conversion on series in datetime data
    if convert:
        date_series1 = pd.to_datetime(date_series1).values.astype('datetime64[D]')
        date_series2 = pd.to_datetime(date_series2).values.astype('datetime64[D]')

    # Building a list with working days calculations between the two dates
    wd_list = [handle_working_day_calc(d1, d2) for d1, d2 in zip(date_series1, date_series2)]

    return wd_list


def indices_of_top_k(arr, k):
    """
    This function selects the top k entries in an array based on its indices

    Parameters
    ----------
    :param arr: numpy array (in practice we will feed it with model feature importance array) [type: np.array]
    :param k: top features integer definition [type: int]

    Return
    ------
    :return: sorted array with filtered input array based on k entries

    Application
    -----------

    """
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])


"""
-----------------------------------
----- 2. CUSTOM TRANSFORMERS ------
   2.1 Pre Processing Pipelines
-----------------------------------
"""


class ColsFormatting(BaseEstimator, TransformerMixin):
    """
    This class applies lower(), strip() and replace() method on a pandas DataFrame object.
    It's not necessary to pass anything as args.

    Return
    ------
    :return: df: pandas DataFrame after cols formatting [type: pd.DataFrame]

    Application
    -----------
    cols_formatter = ColsFormatting()
    df_custom = cols_formatter.fit_transform(df_old)
    """

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        return df


class FeatureSelection(BaseEstimator, TransformerMixin):
    """
    This class filters a dataset based on a set of features passed as argument.

    Parameters
    ----------
    :param features: set of features to be selected on a DataFrame [type: list]

    Return
    ------
    :return: df: pandas DataFrame after filtering attributes [type: pd.DataFrame]

    Application
    -----------
    selector = FeatureSelection(features=model_features)
    df_filtered = selector.fit_transform(df)
    """

    def __init__(self, features):
        self.features = features

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return df[self.features]


class TargetDefinition(BaseEstimator, TransformerMixin):
    """
    This class transform a categorical target column into a numerical one base on a positive_class

    Parameters
    ----------
    :param target_col: reference for the target column on the dataset [type: string]
    :param pos_class: entry reference for positive class in the new target [type: string]
    :param new_target_name: name of the new column created after the target mapping [type: string, default: 'target]

    Return
    ------
    :return: df: pandas DataFrame after target mapping [pd.DataFrame]

    Application
    -----------
    target_prep = TargetDefinition(target_col='class_target', pos_class='Some Category', new_target_name='target')
    df = target_prep.fit_transform(df)
    """

    def __init__(self, target_col, pos_class, new_target_name='target'):
        self.target_col = target_col
        self.pos_class = pos_class
        self.new_target_name = new_target_name

        # Sanity check: new_target_name may differ from target_col
        if self.target_col == self.new_target_name:
            print('[WARNING]')
            print(f'New target column named {self.new_target_name} must differ from raw one named {self.target_col}')

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        # Applying the new target rule based on positive class
        df[self.new_target_name] = df[self.target_col].apply(lambda x: 1 if x == self.pos_class else 0)

        # Dropping the old target column
        return df.drop(self.target_col, axis=1)


class DropDuplicates(BaseEstimator, TransformerMixin):
    """
    This class filters a dataset based on a set of features passed as argument.
    It's not necessary to pass anything as args.

    Return
    ------
    :return: df: pandas DataFrame dropping duplicates [type: pd.DataFrame]

    Application
    -----------
    dup_dropper = DropDuplicates()
    df_nodup = dup_dropper.fit_transform(df)
    """

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return df.drop_duplicates()


class SplitData(BaseEstimator, TransformerMixin):
    """
    This class helps splitting data into training and testing and it can be used at the end of a pre_processing pipe.
    In practice, the class applies the train_test_split() function from sklearn.model_selection module.

    Parameters
    ----------
    :param target: reference of the target feature on the dataset [type: string]
    :param test_size: test_size param of train_test_split() function [type: float, default: .20]
    :param random_state: random_state param of train_test_split() function [type: int, default: 42]

    X_: attribute associated with the features dataset before splitting [1]
    y_: attribute associated with the target array before splitting [1]
        [1] The X_ and y_ attributes are initialized right before splitting and can be retrieved later in the script.

    Return
    ------
    :return: X_train: DataFrame for training data [type: pd.DataFrame]
             X_test: DataFrame for testing data [type: pd.DataFrame]
             y_train: array for training target data [type: np.array]
             y_test: array for testing target data [type: np.array]

    Application
    -----------
    splitter = SplitData(target='target')
    X_train, X_test, y_train, y_test = splitter.fit_transform(df)
    """

    def __init__(self, target, test_size=.20, random_state=42):
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        # Returning X and y attributes (those can be retrieved in the future)
        self.X_ = df.drop(self.target, axis=1)
        self.y_ = df[self.target].values

        return train_test_split(self.X_, self.y_, test_size=self.test_size, random_state=self.random_state)


"""
-----------------------------------
----- 2. CUSTOM TRANSFORMERS ------
    2.2 Preparation Pipelines
-----------------------------------
"""


class DummiesEncoding(BaseEstimator, TransformerMixin):
    """
    This class applies the encoding on categorical data using pandas get_dummies() method. It also retrieves the
    features after the encoding so it can be used further on the script

    Parameters
    ----------
    :param dummy_na: flag that guides the encoding of NaN values on categorical features [type: bool, default: True]

    Return
    ------
    :return: X_dum: Dataframe object (with categorical features) after encoding [type: pd.DataFrame]

    Application
    -----------
    encoder = DummiesEncoding(dummy_na=True)
    X_encoded = encoder.fit_transform(df[cat_features])
    """

    def __init__(self, dummy_na=True):
        self.dummy_na = dummy_na

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # Saving features into class attribute
        self.cat_features_ori = list(X.columns)

        # Applying encoding with pandas get_dummies()
        X_cat_dum = pd.get_dummies(X, dummy_na=self.dummy_na)

        # Joining datasets and dropping original columns before encoding
        X_dum = X.join(X_cat_dum)
        X_dum = X_dum.drop(self.cat_features_ori, axis=1)

        # Retrieving features after encoding
        self.features_after_encoding = list(X_dum.columns)

        return X_dum


class FillNullData(BaseEstimator, TransformerMixin):
    """
    This class fills null data. It's possible to select just some attributes to be filled with different values

    Parameters
    ----------
    :param cols_to_fill: columns to be filled. Leave None if all the columns will be filled [type: list, default: None]
    :param value_fill: value to be filled on the columns [type: int, default: 0]

    Return
    ------
    :return: X: DataFrame object with NaN data filled [type: pd.DataFrame]

    Application
    -----------
    filler = FillNullData(cols_to_fill=['colA', 'colB', 'colC'], value_fill=-999)
    X_filled = filler.fit_transform(X)
    """

    def __init__(self, cols_to_fill=None, value_fill=0):
        self.cols_to_fill = cols_to_fill
        self.value_fill = value_fill

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Filling null data according to passed args
        if self.cols_to_fill is not None:
            X[self.cols_to_fill] = X[self.cols_to_fill].fillna(value=self.value_fill)
            return X
        else:
            return X.fillna(value=self.value_fill)


class DropNullData(BaseEstimator, TransformerMixin):
    """
    This class drops null data. It's possible to select just some attributes to be filled with different values

    Parameters
    ----------
    :param cols_dropna: columns to be filled. Leave None if all the columns will be filled [type: list, default: None]

    Return
    ------
    :return: X: DataFrame object with NaN data filled [type: pd.DataFrame]

    Application
    -----------
    null_dropper = DropNulldata(cols_to_fill=['colA', 'colB', 'colC'], value_fill=-999)
    X = null_dropper.fit_transform(X)
    """

    def __init__(self, cols_dropna=None):
        self.cols_dropna = cols_dropna

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Filling null data according to passed args
        if self.cols_dropna is not None:
            X[self.cols_dropna] = X[self.cols_dropna].dropna()
            return X
        else:
            return X.dropna()


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    """
    This class selects the top k most important features from a trained model

    Parameters
    ----------
    :param feature_importance: array with feature importance given by a trained model [np.array]
    :param k: integer that defines the top features to be filtered from the array [type: int]

    Return
    ------
    :return: pandas DataFrame object filtered by the k important features [pd.DataFrame]

    Application
    -----------
    feature_selector = TopFeatureSelector(feature_importance, k=10)
    X_selected = feature_selector.fit_transform(X)
    """

    def __init__(self, feature_importance, k):
        self.feature_importance = feature_importance
        self.k = k

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, indices_of_top_k(self.feature_importance, self.k)]