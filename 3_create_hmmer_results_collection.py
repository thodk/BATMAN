# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:22:40 2017

@author: thodoris
"""

import core
import os
import sys


main_dir = sys.argv[1]
mongo_db_name = sys.argv[2]
hmm_db_name = sys.argv[3]
n_subfolders = sys.argv[4]

H = core.HmmerResults(mongo_db_name, "training_"+hmm_db_name, hmm_db_name)
H.delete_results_collection()


for i in range(0, int(n_subfolders)):
    tmp_dir = main_dir+str(i)+"/"
    outputs = os.listdir(tmp_dir)
    for f in outputs:
        results = H.read_hmmer_output(tmp_dir+"/"+f)
        H.construct_results_collection(results)

