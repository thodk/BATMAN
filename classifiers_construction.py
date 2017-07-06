# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:05:33 2017

@author: thodoris
"""

import ml_core
import argparse
import multiprocessing
import os
import numpy

parser = argparse.ArgumentParser( prog="BATMAN_CLASSIFIERS_CONSTRUCTION" )
parser.add_argument('-n', type=str, action='store', dest='nprocs', metavar = '',
                    required=False, help="")                    
parser.add_argument('-d', type=str, action='store', dest='directory', metavar = '',
                    required=True, help="")
parser.add_argument('-j', type=str, action='store', dest='job_id', metavar = '',
                    required=True, help="")



args = parser.parse_args()





def worker(i):
    F = open(input_dir+str(i)+".txt", "r")
    ec_numbers = list(j.replace("\n", "") for j in F.readlines())
    r = []
    for ec in ec_numbers:
        C = ml_core.Classificathon(ec, str(args.directory)+"classifiers_for_prokaryotes/")
        C.construct_classifiers()
        r.append(C.df.shape)
    queue.put([i, len(r)])

input_dir = "./"+str(args.directory)+"/batches/"+str(args.job_id)+"/"
# ./classifiers_for_prokaryotes

queue = multiprocessing.Queue()
batch_files = os.listdir(input_dir)
#batch_size = int(numpy.ceil(len(batch_files) / 20.))
#print batch_size

procs = []
for i in range(len(batch_files)):
    p = multiprocessing.Process(target=worker, args=(i, ))
    p.start()
    procs.append(p)


for p in procs:
    p.join()

details = []
for i in range(queue.qsize()):
    details.append(queue.get())

print details

