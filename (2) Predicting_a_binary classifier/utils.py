# import 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def cleaning_data(X,delete_over=10,col_to_change_to_null=0.1):
    """ 
    X: Dataframe that has to be cleaned
    delete_over : columns that have more than delete_over categories will be deleted
    col_to_change_to_null : If a column has more than col_to_change_to_null*100% values in null 
    it will create a new column that will be name column_name + _is_null and would be a boolean
    marking 1 if it the value is null and 0 if not.
    
    """
    categorical_variables = [col for col in  X.columns if str(X[col].dtypes)=='object']
    #numerical_variables = [col for col in X.columns if str(X[col].dtypes)!='object']
    cardinalidad = {}
    for col in categorical_variables:
        cardinalidad[col] = len(list(X[col].unique()))
    # For now we delete categories with more values than..
    delete_over = delete_over
    columns_to_delete = [col for col in categorical_variables if len(list(X[col].unique()))>delete_over ]
    X.drop(columns=columns_to_delete,inplace = True, axis=1)
    
    # We are going to change columns with too many null.
    # We are not gonna delete them, will give them the chance to be important.
    # that means that having or not having the value is what is really important.
    col_to_change_to_null = col_to_change_to_null
    columnas_modificar_por_1 = [col for col in X.columns if X[col].isnull().sum()>int(X.shape[0] * col_to_change_to_null) ]

    for col in columnas_modificar_por_1:
        X[col +str('_is_null')] = 0
        X.loc[(X[col].isnull()),col +str('_is_null')] = 1

    new_columns_null = [str(f"{col}_is_null") for col in columnas_modificar_por_1 ]    
    X.drop(columns=columnas_modificar_por_1, axis=1,inplace=True)
    
    return X



