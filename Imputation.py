import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split

def from_dummies(data, categories, prefix_sep='_'):
    out = data.copy()
    for l in categories:
        cols, labs = [[c.replace(x,"") for c in data.columns if l+prefix_sep in c] for x in ["", l+prefix_sep]]
        out[l] = pd.Categorical(np.array(labs)[np.argmax(data[cols].values, axis=1)])
        out.drop(cols, axis=1, inplace=True)
    return out

def Impute(dataFrame, countOfNullPerRowToFill):
    
    # get dataFrame columns, numeric columns and categorical columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    dataFrameColumns = list(dataFrame.columns)
    dataFrameNumericColumns = list(dataFrame.select_dtypes(include=numerics).columns)
    dataFrameCategoricalColumns = list(set(dataFrameColumns) - set(dataFrameNumericColumns))
    # get dataFrame columns, numeric columns and categorical columns
    
    dummyDataFrame = pd.get_dummies(dataFrame)
    rowsWithoutNull = dataFrame.dropna()
    rowsContainNull = dataFrame[dataFrame.isnull().any(axis=1)] # rows contain one, two or more NULL value in dataFrame
    dummyRowsContainNull = dummyDataFrame[dummyDataFrame.isnull().any(axis=1)] # rows contain one, two or more NULL value in dummyDataFrame
    dummyRowsWithoutNull = dummyDataFrame.dropna()
    
    # ---------------------- regression Imputation for numeric columns -------------------------------------
    for i in range(1, countOfNullPerRowToFill + 1):
        
        #this loop impute rows that have i NULL value in numeric columns.
        #first Impute rows with 1 NULL value in numeric columns.
        #second Impute rows with 2 Null value in numeric columns.
        #And so on to Impute rows with i NULL value in numeric columns.
        
        # this is dummyDataFrame that contains rows with i null value in numeric columns.
        dummyRowsContainINull = dummyRowsContainNull[dummyRowsContainNull.isnull().sum(axis=1) == i]
        # this is dummyDataFrame that contains rows with i null value in numeric columns.
        
        c = dummyRowsWithoutNull.columns
        
        for _in,ro in dummyRowsContainINull.iterrows():# iterate dummyRowsContainINull rows
            desired_columns_X = [] # fill with columns that has null value in (_in) row
            index = 0
            for i,r in dummyRowsContainINull.items(): # iterate dummyRowsContainINull rows' columns
                if r.isnull()[_in]: # if (r) column at (_in) row is NULL
                    desired_columns_X.append(c[index]) # name of column which is null at _in row and add to desired_column list
                index = index + 1
            
            if set(desired_columns_X).issubset(set(dataFrameNumericColumns)) and desired_columns_X:
                # for training
                X = dummyRowsWithoutNull.drop(desired_columns_X,1)
                # for training
                
                # creat target
                y = dummyRowsWithoutNull[desired_columns_X]
                # creat target
                
                # splitting data set
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
                # splitting data set
                
                #create model
                lm = LinearRegression()
                lm.fit(X_train,y_train)
                #create model
                
                #predict null values in rows at dummyRowsContainINull dataFrame
                predictions = lm.predict(pd.DataFrame(dummyRowsContainINull.loc[_in]).T.drop(desired_columns_X,1))
                #predict null values in rows at dummyRowsContainINull dataFrame
                
                # iterate desired_columns_X to fill null values at dummyRowsContainINull
                for des_col_in,des_col_val in enumerate(desired_columns_X):
                    dummyRowsContainINull.at[_in,des_col_val] = predictions[0][des_col_in]
                # iterate desired_columns_X to fill null values at dummyRowsContainINull
                
                # reverse dummyRowsContainINull to dataFrame
                dataFrameFromDummy = from_dummies(pd.DataFrame(dummyRowsContainINull.loc[_in]).T,dataFrameCategoricalColumns)
                # reverse dummyRowsContainINull to dataFrame
                
                rowsWithoutNull = rowsWithoutNull.append(dataFrameFromDummy, sort= False, ignore_index=True)
    # ---------------------- regression for numeric -------------------------------------
    
    # ---------------------- classification for categorical -------------------------------------
    
    # this section impute rows that contain one NULL value in categorical column
    pox = pd.DataFrame()
    pox = rowsWithoutNull
    
    # rows contain 1 null
    rowsContain1Null = rowsContainNull[rowsContainNull.isnull().sum(axis=1) == 1]
    # rows contain 1 null
    
    c = rowsWithoutNull.columns
    
    for _in,ro in rowsContain1Null.iterrows(): # iterate dummyRowsContain1Null rows
        desired_columns_X = [] # fill with columns that has null value in (_in) row
        index = 0
        for i,r in rowsContain1Null.items(): # iterate rowsContain1Null rows' columns
            if r.isnull()[_in]: # if (r) column at (_in) row is NULL
                desired_columns_X.append(c[index]) # name of column which is null at _in row and add to desired_column list
            index = index + 1
        
        if set(desired_columns_X).issubset(set(dataFrameCategoricalColumns)) and desired_columns_X:
            
            X = rowsWithoutNull.drop(desired_columns_X,1)
            y = rowsWithoutNull[desired_columns_X]
            
            X=pd.get_dummies(X)
            
            # if there are another categorical column which are not null, find them and add to remained_column list
            remained_column = []
            dataFrameWithDeletedColumn = pd.DataFrame(rowsContain1Null.loc[_in]).T.drop(desired_columns_X,1)
            for rem_col in dataFrameWithDeletedColumn.columns:
                if rem_col in list(dataFrameCategoricalColumns):
                    remained_column.append(rem_col)
            # if there are another categorical column which are not null, find them and add to remained_column list
            
            # make dummy from rowsContain1Null base on remained_column after deleting categorical column that is null
            dummyRowsContain1Null = pd.get_dummies(dataFrameWithDeletedColumn, columns= remained_column)
            # make dummy from rowsContain1Null base on remained_column after deleting categorical column that is null
            
            # because of one deleted column in dummyRowsContain1Null. count of columns in dummyRowsContain1Null and X are diffrent
            # this section make count of columns dummyRowsContain1Null and columns X the same
            columnsDiff = set(X.columns).difference(set(dummyRowsContain1Null.columns))
            
            for col_diff in list(columnsDiff):
                dummyRowsContain1Null[col_diff] = 0
            # this section make count of columns dummyRowsContain1Null and columns X the same
            
            # splitting data set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
            # splitting data set
            
            #create model            
            lm = LogisticRegression(solver='lbfgs', multi_class='ovr')
            lm.fit(X_train,np.ravel(y_train, order ='C'))
            #create model
            
            #predict null values in rows at dummyRowsContain1Null dataFrame
            predictions = lm.predict(dummyRowsContain1Null)
            #predict null values in rows at dummyRowsContain1Null dataFrame
            
            dataFrameFromDummy = from_dummies(pd.DataFrame(dummyRowsContain1Null.loc[_in]).T,remained_column)
            
            # iterate desired_columns_X to fill null values at dataFrameFromDummy
            for des_col_in,des_col_val in enumerate(desired_columns_X):
                dataFrameFromDummy.at[_in,des_col_val] = predictions[0][des_col_in]
            # iterate desired_columns_X to fill null values at dataFrameFromDummy
            
            pox = pd.concat([pox,dataFrameFromDummy],ignore_index = True, sort=False)
                
    # ---------------------- classification for categorical -------------------------------------
    
    return(pox)