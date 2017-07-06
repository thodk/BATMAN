# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:04:22 2017

@author: thodoris
"""

import core
import sys
import os

mongo_db_name = sys.argv[1]
n_batches = int(sys.argv[2])
n_procs = int(sys.argv[3])
output_dir = sys.argv[4]

tab_col_name = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

FB = core.FastaBatches(mongo_db_name, tab_col_name, n_tasks=n_batches, n_sub_tasks=n_procs)
FB.select_sequences()
FB.split_sequences()
FB.construct_batches(main_directory=output_dir)
