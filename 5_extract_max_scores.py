# -*- coding: utf-8 -*-
"""
Created on Mon May 15 20:28:12 2017

@author: thodoris
"""

import pymongo
import pandas
import numpy 
import sys
import os


mongo_db_name = sys.argv[1]
hmm_db_name = sys.argv[2]
output_dir = sys.argv[3]

# python 5_extract_max_scores.py ecoli_K12 CATH batman_classifiers/ecoli_K12/CATH

client = pymongo.MongoClient()
db = client[mongo_db_name]
collection = db["training_"+hmm_db_name]

scores = {}
res = collection.find()
for i in res:
    scores.setdefault(str(i["domain_id"]), []).append(float(i["score"]))

final = []
for domain_id, scores in scores.items():
    max_score = max(scores)
    final.append([domain_id,max_score])

if os.path.exists(output_dir):
    pass
else:
    os.makedirs(output_dir)

df = pandas.DataFrame(numpy.array(final), columns=["domain_id", "max_score"])

if output_dir[-1] == "/":
    df.to_csv(output_dir+hmm_db_name+"_max_scores.csv", index=0)
else:
    df.to_csv(output_dir+"/"+hmm_db_name+"_max_scores.csv", index=0)
