import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

########### Preliminary SNAP data cleaning ##################################
def SNAP_clean(df_input, reference_columns,target_variable):
    '''
    Will take all the data from the all_df Original dataset and consolidate, clean, and organize the data
    Inputs:
    df
    reference_columns: columns from the reference excel doc that should be input into the dataframe used in this notebook. Multiple columns need to be in square brackets []
    target_variable: string object y dependent variable
    '''
    # Re-coding survey administrative data
    df_input = df_input.replace([-996, -997,-998], [0 ,-1, -1])
    
    # Removing any rows where values in the target variable are undefined
    df_input = df_input.loc[df_input[target_variable]>=0,:]     #only taking non-imputted values
    
    #Converting Float64 variables to Float32
    float64_cols = list(df_input.select_dtypes(include='float64'))
    df_input[float64_cols] = df_input[float64_cols].astype('float32')
    
    #converting all number columns to numerical
    df_output = pd.get_dummies(df_input, drop_first = True)
    
    x = df_output[df_output.columns.intersection(reference_columns)]
    y = df_output[target_variable]
    df = pd.concat([x, y], axis = 1)
    
    #Finding null columns and number of rows
    null_df = pd.DataFrame(df.isnull().sum(), columns = ['nulls'])
    null_df = null_df[null_df['nulls']>0]
    return x,y,df, null_df


########## Deleting Null Values in Row or column as specified ######################
def delete_nans(df, target_variable, select_axis):
    '''
    select_axis = 0 to delete rows with columns
    select_axis = 1 to delete columns with nans
    '''
    df.dropna(axis = select_axis, inplace = True)
    x_cleaned = df.drop(columns = target_variable)
    y_cleaned = df[target_variable]
    return x_cleaned, y_cleaned,df


######## Decision Tree Feature Selection##########3
def feature_selection_dt(model, new_X, new_y):
    '''
    Decision Tree Model used for preliminary feature selection
    Input:
    X dataset
    Y target variable
    Iteration number which will become the key name in the dictionary
    '''
    variables_remove = []
    model.fit(new_X, new_y)
    feature_importance = pd.DataFrame(zip(new_X.columns,model.feature_importances_), columns = ['feature','importance'])  
    zero_sig = feature_importance['feature'][feature_importance['importance']==0]
    [variables_remove.append(x) for x in zero_sig]
    print(f'{new_X.shape[1]} Columns Originally \n{len(zero_sig)} Discard Variables\n')
    if len(zero_sig)==0:
        display(feature_importance.sort_values('importance', ascending = False))
    return variables_remove
