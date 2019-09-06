#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 23:11:47 2019

@author: mmann
"""

from itertools import product
import pandas as pd
import numpy as np


def expand_grid(dictionary):
    '''
    Creates full set of longitudinal data from dictionary containing all
    unique values of variables
    
    :param dictionary: dictionary where key is column name and values are all unique used in expansion
   
    :return: cartesian product or expanded grid 
    '''
    
    return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())
   
    
def count_missing_by(df, column, group):
    '''
    counts number of missing values in a group 
    
    :param df: dataframe 
    :param column: text column name to count missing values in  
    :param group: group_by column name 

    :return: cartesian product or expanded grid 
    '''
    
    return(df[column].isnull().groupby(df[group]).sum()\
                            .astype(int).reset_index(name='count'))
    


def missing_group(df, group, column, min_missing = 2000):
    '''
    Finds groups in df that have a min number of missing values 
    
    :param df: dataframe 
    :param group: group_by column name 
    :param column: text column name to count missing values in  
    :param min_missing: return groups with min of this # of missing values

    :return: group ids meeting min_missing criteria
    '''
    
    out = df.groupby(group).filter(lambda g: g[column]\
                    .isnull().sum() > min_missing)[group].unique()
    
    return( out )


def buildLaggedFeatures(s,lag=2,dropna=True):
    
    '''
    Builds a new DataFrame to facilitate regressing over all possible lagged features
    
    '''
    
    if type(s) is pd.DataFrame:
        new_dict={}
        for col_name in s:
            new_dict[col_name]=s[col_name]
            # create lagged Series
            for l in range(1,lag+1):
                new_dict['%s_lag%d' %(col_name,l)]=s[col_name].shift(l)
        res=pd.DataFrame(new_dict,index=s.index)
    
    elif type(s) is pd.Series:
        the_range=range(lag+1)
        res=pd.concat([s.shift(i) for i in the_range],axis=1)
        res.columns=['lag_%d' %i for i in the_range]
    else:
        print('Only works for DataFrame or Series')
        return(None)
    if dropna:
        return(res.dropna())
    else:
        return(res) 