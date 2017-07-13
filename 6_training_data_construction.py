# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 23:25:35 2017

@author: thodoris
"""

import core
import ml_core
import sys
import multiprocessing
import os
import operator
import pymongo

main_dir = sys.argv[1]
mongo_db_name = sys.argv[2]
hmm_db_name = sys.argv[3]

# python 6_training_data_construction.py batman_classifiers/prokaryotes/PFAM/ prokaryotes PFAM

client = pymongo.MongoClient()
db = client[mongo_db_name]
collection = db["ec_numbers"]
results = collection.find({},{"ec_number":1})
ecs = []
for entry in results:
    ecs.append(entry["ec_number"])

ecs = sorted(list(set(ecs)))
lengths = map(lambda ec: str(len(ec.replace("-", "").replace(".", ""))), ecs)

ec_tuples = zip(ecs, lengths)
ec_tuples = sorted(ec_tuples, key=operator.itemgetter(1), reverse=True)


ec3 = list(ec[0] for ec in ec_tuples if ec[0].startswith("3."))
a = len(ec3)
i=0
#ec3 = ["1.-.-.-", "4.-.-.-"]
for ec_number in ec3:
    if ec_number.count("-") == 3:
        continue
    i+=1
    if ec_number == "3.4.11.10":
    	print ec_number, str(i), "/", str(a)
    	TD = ml_core.TrainingDatathon(ec_number, mongo_db_name, hmm_db_name, main_dir)

