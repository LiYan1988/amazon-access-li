# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 08:36:08 2016

@author: MULTIPOS
"""

import numpy as np
import pandas as pd

def findMultiplicity(df):
    '''
        This finds the multiplicity of different 
        columns. If a multiplicity is 1, it is going to
        be safe to just neglect one of the categorical 
        variables
    '''

    # Find the multiplicity ....
    mult = {}
    for c in df.columns:
        temp = df.pivot_table( index=c, 
                                  aggfunc=lambda x:len(x.unique()),
                                  fill_value=0).apply(np.max)
        mult[c] = temp

    mult1 = pd.DataFrame(mult)
    mult1 = mult1[ sorted(mult1.columns) ]

    return mult1