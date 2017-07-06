# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:28:32 2017

@author: thodoris
"""
import sys
import core

input_tab_file = sys.argv[1]
db_name = sys.argv[2]


ec_col_name = "ec_numbers"
tab_col_name= "data"


UPM = core.UniprotMongo(db_name, tab_col_name, ec_col_name)

'''
UPM.update includes the deletion of tab and ec_numbers collection
and the construction of new collection based on input file
'''
UPM.update(input_tab_file)
