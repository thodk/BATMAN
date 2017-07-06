# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 18:12:24 2016

@author: thodoris
"""

import pymongo
import pandas
import os
import math
import numpy
import operator
import copy
import cPickle
import subprocess
from sklearn import linear_model
from sklearn import tree
from sklearn import model_selection
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AffinityPropagation
from imblearn import over_sampling
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

numpy.random.seed(1234)

class MongoWorker(object):
    '''
    MongoWorker class provides all the appropriate methods to connect with MongoDB
    client and construct or delete a collection, given the respective database,
    collection name, input list of documents and indices.
    
    It is used as an inherited class from all the others that communicate with 
    MongoDB collections.
    '''
    def __init__(self):
        pass

    def connect_to_mongo(self, db, collection):
        client = pymongo.MongoClient()
        tmp_db = client[db]        
        tmp_collection = tmp_db[collection]
        return client, tmp_db, tmp_collection

    def delete_collection(self, db, collection):
        client, tmp_db, tmp_collection = self.connect_to_mongo(db, collection)
        tmp_collection.delete_many({})      
        client.close()

    def construct_collection(self, db, collection, data, indices=[]):
        client, tmp_db, tmp_collection = self.connect_to_mongo(db, collection)
        tmp_collection.insert_many(data)
        for index in indices:
            tmp_collection.create_index(index)
        client.close()



class UniprotMongo(MongoWorker):
  
  
    def __init__(self, db, tab_col, ec_col):
        self.db = db
        self.tab_col = tab_col
        self.ec_col = ec_col       

    def semicolon_strings(self, tmp_dict, key, sep):
        try:
            key_string = tmp_dict[key]
            try:
                math.isnan(key_string)
                return []
            except TypeError:
                key_fields = key_string.split(sep)
                if key_fields[-1] == "":
                    key_fields.remove(key_fields[-1])
                return key_fields
        except KeyError:
            return []   
    
    def construct_tab_data(self, input_file=None):
        if input_file==None:
            return []
        if not os.path.isfile(input_file):
            return []
        output = [] # list of dictionaries
        F = pandas.read_table(input_file)
        F.columns = list(a.lower().replace(" ", "_") for a in F.columns.values)
        F.columns = list(a.translate(None, "()") for a in F.columns.values)
        for i in range(F.shape[0]):
            print i
            tmp_dict = F.ix[i,].to_dict() # Series to dict
            try:
                names_string = tmp_dict["gene_names"]
                try:
                    math.isnan(names_string)
                    basic = None
                    alternatives = []
                except TypeError:
                    names = names_string.split(" ")
                    basic = names[0]
                    try:
                        alternatives = names[1:]
                    except IndexError:
                        alternatives = []
                tmp_dict["gene_names"] = {"recommended":basic, "alternatives":alternatives}
            except KeyError:
                pass
            tmp_dict["ec_number"] = self.semicolon_strings(tmp_dict, "ec_number", "; ")         
            tmp_dict["cross-reference_pfam"] = self.semicolon_strings(tmp_dict, "cross-reference_pfam", ";") 
            tmp_dict["cross-reference_supfam"] = self.semicolon_strings(tmp_dict, "cross-reference_supfam", ";")
            tmp_dict["cross-reference_gene3d"] = self.semicolon_strings(tmp_dict, "cross-reference_gene3d", ";")
            output.append(tmp_dict)
        return output

    def construct_ec_data(self, tab_data):
        ec_data = []
        for entry in tab_data:
            uniprot_id = entry["entry"]
            ec_numbers = entry["ec_number"]
            ec_numbers_final = []
            ec_numbers_majors = []
            for major_ec in ec_numbers:
                ec_numbers_majors.append(major_ec)
                tmp_fields = major_ec.split('.')
                for i in range(1,5):
                    if tmp_fields[i-1] == '-':
                        continue
                    else:
                        new_ec_fields = tmp_fields[0:i] + list("-" for j in range(i+1,5))
                        ec_numbers_final.append(".".join(new_ec_fields))
            for ec in ec_numbers_final:
                if ec in ec_numbers_majors:
                    ec_data.append({"entry":uniprot_id, "ec_number":ec, "major":1})
                else:
                    ec_data.append({"entry":uniprot_id, "ec_number":ec}) 
        return ec_data

    def delete_tab_collection(self):
        self.delete_collection(self.db, self.tab_col)
        
    def delete_ec_collection(self):
        self.delete_collection(self.db, self.ec_col)

    def construct_tab_collection(self, tab_data):
        self.construct_collection(self.db, self.tab_col, tab_data, 
                                      indices=['entry'])
        
    def construct_ec_collection(self, ec_data):
        self.construct_collection(self.db, self.ec_col, ec_data, 
                                  indices=['entry', 'ec_number'])

    def append(self, input_file):
        tab_data = self.construct_tab_data(input_file)
        self.construct_collection(self.db, self.tab_col, tab_data, 
                                      indices=['entry'])
        ec_data = self.construct_ec_data(tab_data)
        self.construct_collection(self.db, self.ec_col, ec_data, 
                                  indices=['entry', 'ec_number'])        

    def update(self, input_file):
        self.delete_tab_collection()
        self.delete_ec_collection()
        tab_data = self.construct_tab_data(input_file)
        self.construct_collection(self.db, self.tab_col, tab_data, 
                                      indices=['entry'])
        ec_data = self.construct_ec_data(tab_data)
        self.construct_collection(self.db, self.ec_col, ec_data, 
                                  indices=['entry', 'ec_number'])

    def extract_fasta_data(self):
        client, db, collection = self.connect_to_mongo(self.db, self.tab_col)
        results = collection.find()
        tmp_data = []
        for res in results:
            tmp_data.append({"entry":res["entry"], "sequence":res["sequence"]})
        return tmp_data

    def write_fasta(self, fasta_data, file_name):
        i=1
        F = open(file_name, "w")
        for entry in fasta_data:
            if i == 1:
                F.write(">"+entry["entry"]+"\n"+entry["sequence"])
            else:
                F.write("\n>"+entry["entry"]+"\n"+entry["sequence"])
            i+=1
        F.close()



 

    






class FastaBatches(MongoWorker):
    '''
    Given a UniprotMongo created database, with the respective tab collection
    and ec_numbers collects, that class is able to extract fasta files scheduled
    in tasks-subtasks sets for parallel analysis
    '''
    def __init__(self, db, tab_col, ec_col=None, name=None, n_tasks=1, n_sub_tasks=20):
        self.db = db
        self.tab_col = tab_col
        self.ec_col = ec_col
        self.name = name
        self.n_tasks = n_tasks
        self.n_sub_tasks = n_sub_tasks
        self.data = []
        self.mapping = {}
        for i in range(self.n_tasks):
            for j in range(self.n_sub_tasks):
                self.mapping.setdefault(str(i), {}).setdefault(str(j), [])

    def select_sequences(self, N=None, ec_numbers=None):
        if ec_numbers == None:
            tmp_client, tmp_db, tmp_tab_col = self.connect_to_mongo(self.db, self.tab_col)
            results = tmp_tab_col.find({}, {'entry':1, 'sequence':1})
            for result in results:
                self.data.append({"entry":result["entry"], "sequence":result["sequence"]})
        else:
            tmp_client, tmp_db, tmp_ec_col = self.connect_to_mongo(self.db, self.ec_col)
            if type(ec_numbers) == list:
                ec_results = tmp_ec_col.find({'ec_number':{'$in':ec_numbers}}, {'entry':1})
            else:
                ec_results = tmp_ec_col.find({'ec_number':ec_numbers}, {'entry':1})
            selected_entries = []
            for entry in ec_results:
                selected_entries.append(entry["entry"])
            tmp_client.close()
            
            tmp_client, tmp_db, tmp_tab_col = self.connect_to_mongo(self.db, self.tab_col)
            for entry in selected_entries:
                results = tmp_tab_col.find({"entry" : entry} , {'entry':1, 'sequence':1}) 
                for result in results:
                    self.data.append({"entry":result["entry"], "sequence":result["sequence"]})
            tmp_client.close()

        if (type(N) == int and len(self.data) > N):
            numpy.random.seed(1234)
            self.data = numpy.random.choice(self.data, N, replace=False)
        
        return self.data


    def fasta_from_tab(self, input_file, sep):
        df = pandas.read_csv(input_file, sep=sep)
        for row in range(df.shape[0]):
            self.data.append({"entry":row["Entry"], "sequence":row["Sequence"]})
        print len(self.data)

    def split_sequences(self):
        numpy.random.seed(1234)
        length = len(self.data)
        batch = int(numpy.math.floor(length/float(self.n_tasks)))
        sub_batch = int(numpy.math.floor(batch/float(self.n_sub_tasks)))
        for i in range(self.n_tasks):
            if i == range(self.n_tasks)[-1]:
                batch_seqs = self.data[i*batch:]    
            else:
                batch_seqs = self.data[i*batch:(i+1)*batch]
    
            for j in range(self.n_sub_tasks):
                if j == range(self.n_sub_tasks)[-1]:
                    sub_batch_seqs = batch_seqs[j*sub_batch:]   
                else:
                    sub_batch_seqs = batch_seqs[j*sub_batch:(j+1)*sub_batch]
                print i,len(batch_seqs),j, len(sub_batch_seqs)
                
                self.mapping[str(i)][str(j)] = sub_batch_seqs

    def construct_batches(self, main_directory="./"):
        if self.name == None:
            self.name = str(self.n_tasks) + "_" + str(self.n_sub_tasks)
        dir_name = main_directory+self.name

        if os.path.exists(str(dir_name)+"/"):
            pass
        else:
            command = subprocess.Popen("mkdir "+str(dir_name), stdout=subprocess.PIPE, shell=True)
            command.communicate(str(dir_name))        
     
        for i in self.mapping.keys():
            directory = str(dir_name)+"/"+i+"/"
            command = subprocess.Popen("mkdir "+directory, stdout=subprocess.PIPE, shell=True)
            command.communicate()
            batches = self.mapping[i]
            for j, tmp_list in batches.items():
                F = open(directory+str(j)+'.fa', 'w')
                c=1
                for entry in tmp_list:
                    if c == 1:
                        F.write(">"+entry["entry"]+"\n"+entry["sequence"])
                    else:
                        F.write("\n>"+entry["entry"]+"\n"+entry["sequence"])
                    c+=1
                F.close()                 

 
 
 


class HmmerResults(MongoWorker):
    

    def __init__(self, db, collection, hmm_database):
        self.res_db = db
        self.res_col = collection
        self.hmm_database = hmm_database

    def read_hmmer_output(self, input_file):
        F = open(input_file, "r")
        results_list = []
        try:
            next(F)
        except StopIteration:
            print "ERROR: Hmmer results file is empty."
            return []
        header = next(F)
        next(F)
        header = header.replace("# ", "")
        header = header.replace("target name ", "query_id")
        header = header.replace("accession ", "domain_id")
        header = header.replace("query name ", "domain_id")
        names = list(a.lower() for a in header.split())
        if self.hmm_database == "CATH":
            indexes = [2,0,4,5]
        else:
            indexes = [3,0,4,5]
        for line in F:
            if line.startswith("#"):
                continue
            fields = line.split()
            if float(fields[4]) > 0.01:
                continue
            else:
                tmp_dict = {}
                for i in indexes:
                    if self.hmm_database == "CATH" and i==2:
                        tmp = fields[i].split("|")[2]
                        tmp = tmp.split("/")[0]
                        fields[i] = tmp
                    tmp_dict.update({names[i]: fields[i]})
                results_list.append(tmp_dict)
        F.close()
        return results_list       

    def delete_results_collection(self):
        self.delete_collection(self.res_db, self.res_col)
        
    def construct_results_collection(self, results_data):
        self.construct_collection(self.res_db, self.res_col, results_data, 
                                  indices=['query_id', 'domain_id'])     



def find_ec_number_linkage(ec):
    output = []
    tmp_fields = ec.split('.')
    for i in range(1,5):
        if tmp_fields[i-1] == '-':
            continue
        else:
            new_ec_fields = tmp_fields[0:i] + list("-" for j in range(i+1,5))
            output.append(".".join(new_ec_fields))    
    output = sorted(output)
    return output


def remove_duplicates_from_df(df):
    
    mat = df.as_matrix()
    nns = NearestNeighbors(n_neighbors=mat.shape[0]).fit(mat)
    distances, indices = nns.kneighbors(mat)

    distances_df = pandas.DataFrame(numpy.around(distances,2))
    indices_df = pandas.DataFrame(indices.astype(numpy.int))
    matches = {}
    tmp_list = range(df.shape[0])
    while len(tmp_list)>0:
        i=tmp_list[0]
        tmp_dist = distances_df.ix[i,:]
        similar_len = len(list(tmp_dist[tmp_dist==0]))
        tmp_ind = sorted(list(indices_df.ix[i,range(similar_len)]))
        key = tmp_ind[0]
        values = tmp_ind[1:]
        matches.update({key:values})
        for j in list(tmp_ind):
            tmp_list.remove(j)
            
    final_df = df.ix[matches.keys(),:]
    return final_df





def get_specific_class_df(dataframe, class_ind):
    df = dataframe.ix[dataframe['class'].str.startswith("no")]
    if class_ind == "positive_class":
        neg = list(df.index)
        total = list(dataframe.index)
        pos = list(set(total) - set(neg))
        df = dataframe.loc[pos]
    else:
        pass
    df = df.ix[:, df.columns != 'class']
    return df

def get_non_zero_columns(matrix):
    non_zero_columns = []
    for i in range(matrix.shape[1]):
        if not all(matrix[:,i]==0):
            non_zero_columns.append(i)
    return non_zero_columns


def create_binary_matrix(matrix):
    array = []
    for i in range(matrix.shape[0]):
        new_row = list(1 if j>0 else 0 for j in matrix[i,:])
        array.append(new_row)
    return numpy.array(array)


def set_binary_clusters(binary_matrix):
    distances = list(pdist(binary_matrix, 'minkowski', p=1))
    matches = {}; dels = []; c=0
    for row_index in range(binary_matrix.shape[0]):
        try:
            dels.index(row_index)
            c = c + len(range(binary_matrix.shape[0]))-row_index-1
            if c == len(distances)+1:
                break
        except ValueError:
            matches.setdefault(row_index, []).append(row_index)
            row_index,
            for j in range(row_index+1, binary_matrix.shape[0]):
                if distances[c] == 0:
                    matches.setdefault(row_index, []).append(j)
                    dels.append(j)
                c+=1
    return matches.values()


def affinity_propagation_clustering(matrix):    
    aff_obj = AffinityPropagation(preference=-50).fit(matrix)
    return aff_obj


def binary_clusters_metanalysis(binary_clusters, matrix):
    clusters = []
    for batch in binary_clusters: # for instance: [[0], [1], [2, 3]]
        if len(batch) > 1:
            aff_obj = affinity_propagation_clustering(matrix[batch])
            aff_labels = aff_obj.labels_
            aff_clusters = {}
            for i in range(len(aff_labels)):
                try:
                    aff_clusters.setdefault(aff_labels[i], []).append(batch[i])
                except TypeError:
                    aff_clusters.setdefault(0, []).append(batch[i])
            for aff_cluster, members in aff_clusters.items():
                clusters.append(members)
        else:
            clusters.append(batch)    
    return clusters



def adaptive_clustering(matrix):
    binary_matrix = create_binary_matrix(matrix)  
    binary_clusters = set_binary_clusters(binary_matrix)
    final_clusters = binary_clusters_metanalysis(binary_clusters, matrix)   
    return final_clusters



def noise_addition(value, noise):
    if value + noise < 0:
        value = round(value + abs(noise), 2)
    else:
        value = round(value + noise, 2)
    return value


def samples_generation(matrix, N_new, fix=False, loc=0.0, scale=1.0):    
    numpy.random.seed(1234)
    non_zero_columns = get_non_zero_columns(matrix)   
    N = matrix.shape[0]
    columns_count = matrix.shape[1]
    while N<N_new:
        new_sample = [0]*columns_count
        if fix == False: 
            for i in non_zero_columns:
                mean_value = numpy.mean(matrix[:,i])
                std_value = numpy.std(matrix[:,i])
                new_value = numpy.random.normal(mean_value, std_value)
                if std_value <= 2.0:
                    noise = numpy.random.normal(scale=scale)
                    new_value = noise_addition(new_value, noise)
                elif std_value <= 3.0:
                    noise = numpy.random.normal(scale=std_value)
                    new_value = noise_addition(new_value, noise)
                else:
                    noise = numpy.random.normal(scale=std_value/2.)
                    new_value = noise_addition(new_value, noise)
                new_value = min([100, new_value])
                new_value = max([0, new_value])
                new_sample[i] = round(new_value, 2)
            array = numpy.transpose(numpy.array(new_sample))
            matrix = numpy.vstack((matrix, array))
            N+=1
        else:
            for i in non_zero_columns:
                mean_value = numpy.mean(matrix[:,i])
                std_value = numpy.std(matrix[:,i])
                noise = numpy.random.normal(mean_value, std_value)
                new_value = noise_addition(0, noise)
                noise = numpy.random.normal(scale=scale)
                new_value = noise_addition(new_value, noise)
                new_sample[i] = round(new_value, 2)
            array = numpy.transpose(numpy.array(new_sample))
            matrix = numpy.vstack((matrix, array))
            N+=1            
    return matrix


def class_imbalance_oversampling(df, label, minority_class):
    if minority_class == "positive_class":
        si = "pos_"
    else:
        si = "neg_"
    features = list(df.columns.values)
    class_df = get_specific_class_df(df, minority_class)
    rownames = list(class_df.index)
    other_class = list(set(list(df.index)) - set(rownames))
    final = df.ix[other_class]

    matrix = class_df.as_matrix().astype(numpy.float)
    final_clusters = adaptive_clustering(matrix)
    
    batch_count = int(float(df.shape[0] - class_df.shape[0]) / len(final_clusters))
    j = class_df.shape[0]
    for indices in final_clusters:
        sub_matrix = matrix[indices]
        new_sub_matrix = samples_generation(sub_matrix, batch_count)
        existed_rownames = list(list(class_df.index)[i] for i in indices)
        generated_rownames = list(si+str(i+j) for i in range(batch_count-len(indices)))
        j = j + len(generated_rownames)
        rownames = existed_rownames + generated_rownames
        tmp_class = list(df.ix[existed_rownames]['class'])[0]
        tmp_df = pandas.DataFrame(new_sub_matrix, index=rownames, 
                                  columns=class_df.columns)
        tmp_df['class'] = tmp_class
        final = final.append(tmp_df)

    features = list(final.columns.values)
    features.remove('class')
    features.insert(0, 'class')
    final = final[features]
    return final
    


def min_majority_size(minority_size):
    if minority_size < 5:
        return 200.
    else:
        return 200.

          

def execute_adasyn(df, label, minority_class):

    def worker(array,x):
        return list(0 if i<10 else round(i,2) for i in list(array[x]))

    
    X_df = df.ix[:, df.columns != label]
    features = X_df.columns
    y_df = df[[label]]

    X_mat = X_df.as_matrix()            
    y_mat = y_df.as_matrix().ravel()  
            
    adasyn_obj = over_sampling.ADASYN(k=30)           
    X_mat_new, y_mat_new = adasyn_obj.fit_sample(X_mat,y_mat)

    new_examples_count = X_mat_new.shape[0] - X_mat.shape[0]
    if minority_class == "positive_class":
        new_rownames = list("pos_"+str(i) for i in range(new_examples_count))
    else:
        new_rownames = list("neg_"+str(i) for i in range(new_examples_count))        
    X_mat_new_examples = X_mat_new[X_mat.shape[0]:]   
    X_mat_new_examples = numpy.array(map(lambda x: worker(X_mat_new_examples,x), range(new_examples_count)))
    
    X_df_new = pandas.DataFrame(X_mat_new_examples, index=new_rownames, columns=features)
    y_df_new = pandas.DataFrame(y_mat_new[X_mat.shape[0]:], index=new_rownames, columns=['class'])
    new_examples_df = pandas.concat([y_df_new, X_df_new], axis=1)
    
    df = pandas.concat([df, new_examples_df], axis=0)
    return df

   



class ClassifierFramework(MongoWorker):


    def __init__(self, ec_number, hmm_db_name, mongo_db, mongo_tab_col, 
                 mongo_ec_col, mongo_res_col, main_dir):
        # Mongo variables
        self.SP = UniprotMongo(mongo_db, mongo_tab_col, mongo_ec_col)
        self.mongo_res_col = mongo_res_col

        # EC number and classes
        self.ec_number = ec_number
        self.ec_numbers_linkage = find_ec_number_linkage(ec_number)
        self.upper_ec_numbers = self.ec_numbers_linkage[:-1]
        self.positive_class = ec_number
        self.negative_class = "no_"+ec_number

        # Examples & features dictionaries
        self.examples = {"positives":[], "negatives":[], "related_negatives":{}, "final_negatives":[]}
        self.features = {"positives":[], "negatives":[], "final_negatives":[]}
        self.features_pool = {}
        self.examples_pool = {}
        self.classes = {}
        self.generated_data = {"positives":None, "negatives":None}
        
        # max_scores
        df = pandas.read_csv(main_dir + hmm_db_name + "_max_scores.csv", index_col=0)        
        scores = map(lambda x: round(x,1), df.to_dict()['max_score'].values())
        domains = df.to_dict()['max_score'].keys()
        tuples = zip(domains, scores)
        self.max_scores_dict = dict((x, y) for x, y in tuples)

        # Create directory and log files
        specific_directory = "/".join(self.ec_numbers_linkage)+"/"
        self.directory = main_dir + specific_directory
        if os.path.exists(self.directory):
            pass
        else:
            os.makedirs(self.directory)
        self.file_name = self.directory+"data_info.csv"
        F = open(self.file_name, 'w')
        F.write("positive class,"+self.positive_class+"\n")
        F.write("negative class,"+self.negative_class+"\n")
        F.close()

    def get_pos_examples(self):
        client, db, collection = self.SP.connect_to_mongo(self.SP.db, 
                                                          self.SP.ec_col)
        #results = collection.find({'ec_number':{'$in':ec_numbers}})
        results = collection.find({'ec_number':self.ec_number})
        for entry in results:
            self.examples.setdefault("positives", []).append(entry["entry"])
        client.close()
        
        F = open(self.file_name, 'a')
        F.write("positives_background,"+str(len(self.examples["positives"]))+"\n")
        F.close()
        return self.examples["positives"]
    
    def get_pos_features(self):
        tmp_client, tmp_db, tmp_col = self.connect_to_mongo(self.SP.db, 
                                                            self.mongo_res_col)
        results = tmp_col.find({'query_id':{'$in':self.examples["positives"]}})
        tmp_pos_examples = []
        tmp_pos_features = []
        for entry in results:
            tmp_pos_features.append(entry["domain_id"])
            tmp_pos_examples.append(entry["query_id"])
            domain = entry["domain_id"]
            max_score = round(float(self.max_scores_dict[domain]),1)
            abs_score = float(entry["score"])
            normalized_score = 100*round(abs_score/max_score, 4)
            self.features_pool.setdefault(entry["domain_id"], []).append(entry["query_id"])
            self.examples_pool.setdefault(entry["query_id"], 
                                          {}).update({domain : normalized_score})
                           
        self.examples["positives"] = list(set(tmp_pos_examples))[:]
        self.features["positives"] = list(set(tmp_pos_features))[:]

        F = open(self.file_name, 'a')
        F.write("positives_annotated,"+str(len(self.examples["positives"]))+"\n")
        F.write("features_from_positives,"+str(len(self.features["positives"]))+"\n")
        F.close()
        return self.features["positives"]

    def get_neg_examples(self):
        tmp_client, tmp_db, tmp_col = self.connect_to_mongo(self.SP.db, 
                                                            self.mongo_res_col)
        results = tmp_col.find({'domain_id':{'$in':self.features["positives"]}}) # get positives & negatives
        tmp_neg_examples = []
        for entry in results:
            tmp_neg_examples.append(entry["query_id"])     
        tmp_neg_examples = list(set(tmp_neg_examples))[:]
        for pos_example in self.examples["positives"]: # remove positives
            tmp_neg_examples.remove(pos_example)
        self.examples["negatives"] = list(set(tmp_neg_examples))[:]
        
        F = open(self.file_name, 'a')
        F.write("negatives_annotated,"+str(len(self.examples["negatives"]))+"\n")
        F.close()
        return self.examples["negatives"]

    def get_upper_classes_neg_examples(self):      
        if len(self.upper_ec_numbers) == 0:
            F = open(self.file_name, 'a')
            for ec in sorted(self.upper_ec_numbers, reverse=True):
                F.write("negatives_annotated_"+ec+","+str(len(self.examples["related_negatives"][ec]))+"\n")
                F.close()            
            return

        client, db, collection = self.SP.connect_to_mongo(self.SP.db, self.SP.ec_col)
        tmp_dict = {}
        for ec_number in self.upper_ec_numbers:
            self.examples["related_negatives"].setdefault(ec_number, [])
            results = collection.find({'ec_number':ec_number})
            for entry in results:
                tmp_dict.setdefault(entry["entry"], []).append(ec_number)
        client.close() 
        
        for example in self.examples["negatives"]:
            try:
                tmp_ecs = tmp_dict[example]
                splitted_ecs = list(ec.split(".") for ec in tmp_ecs)
                length = len(splitted_ecs)
                tmp_dash_counts = list(splitted_ecs[i].count('-')*[tmp_ecs[i]] for i in range(length))
                tmp_dash_counts.sort(key = lambda i: len(i))
                final_ec = tmp_dash_counts[0][0]
                self.examples["related_negatives"].setdefault(final_ec, []).append(example)
            except KeyError:
                pass     

        F = open(self.file_name, 'a')
        for ec in sorted(self.upper_ec_numbers, reverse=True):
            F.write("negatives_annotated_"+ec+","+str(len(self.examples["related_negatives"][ec]))+"\n")
        F.close()
        return

    def finalize_negatives_population(self):
        numpy.random.seed(1234)
        final_list = []
        positives_count = float(len(self.examples["positives"]))
        c = len(self.upper_ec_numbers) 
        if c == 0:
            final_list = self.examples["negatives"][:]
        else:
            while c > 0:
                try:
                    ratio = numpy.log2(len(final_list)) - numpy.log2(positives_count)
                except ValueError:
                    ratio = 0
                if (len(final_list) < min_majority_size(positives_count) or ratio<0.32):
                    ec = sorted(self.upper_ec_numbers)[c-1]
                    final_list = self.examples["related_negatives"][ec] + final_list
                    c-=1
                else:
                    break
        try:
            ratio = numpy.math.log(len(final_list)/positives_count, 2)
        except ValueError:
            ratio = 0

        # TRY RANDOM SELECTION                     
        if (len(final_list) > min_majority_size(positives_count) and ratio>0.32):
            self.examples["final_negatives"] = final_list[:]
            F = open(self.file_name, 'a')
            F.write("add_random_selected_negatives,False\n")
        else:
            add_examples = numpy.random.choice(self.examples["negatives"], int(min_majority_size(positives_count)))
            final_list = list(add_examples) + list(final_list[:])
            final_list = list(set(final_list))
            self.examples["final_negatives"] = final_list[:]
            F = open(self.file_name, 'a')
            F.write("add_random_selected_negatives,True\n")

        F.write("negatives_final,"+str(len(self.examples["final_negatives"]))+"\n")
        F.close()


    def generate_negatives(self):
        
        def generate_artificials(N, j):
            new_negatives = []
            rownames = []
            examples = self.examples["positives"][:]
            rows_data = []
            # create dataset only with positive examples            
            for tmp_example in examples:
                tmp_data = []
                for tmp_feature in features:
                    try:
                        tmp_data.append(self.examples_pool[tmp_example][tmp_feature])
                    except:
                        tmp_data.append(0)
                rows_data.append(tmp_data)
            matrix = numpy.array(rows_data)
            # cluster positive examples
            final_clusters = adaptive_clustering(matrix)
            batch_count = int(N/len(final_clusters))
            for indices in final_clusters:
                sub_matrix = matrix[indices]
                for e in range(batch_count):
                    new_sample = []
                    for i in range(sub_matrix.shape[1]):
                        mean = numpy.mean(sub_matrix[:,i])
                        std = numpy.std(sub_matrix[:,i])
                        if std < 1.0:
                            std = 1.0
                        max_lim = min([100, mean ])
                        min_lim = max([0, mean ])
                        if max_lim > 96:
                            x = []
                        else:    
                            x = list(numpy.random.uniform(max_lim, 100, 10))
                        y = list(numpy.random.uniform(0, min_lim, 10)) 
                        new_value = round(numpy.random.choice(x+y),2)
                        new_sample.append(new_value)
                    new_negatives.append(new_sample)
                    rownames.append("neg_"+str(j))
                    j+=1
            return rownames, new_negatives
        
        
        def clone_negatives(N, j):
            new_negatives = []
            self.examples["final_negatives"] = self.examples["negatives"][:]
            self.get_neg_features() # in order to take the scores
            examples = self.examples["negatives"][:]
            rownames = examples[:]
            # create a dataset only with existed negatives
            rows_data = []
            for tmp_example in examples:
                tmp_data = []
                for tmp_feature in features:
                    try:
                        tmp_data.append(self.examples_pool[tmp_example][tmp_feature])
                    except:
                        tmp_data.append(0)
                rows_data.append(tmp_data)
            matrix = numpy.array(rows_data)
            final_clusters = adaptive_clustering(matrix)
            batch_count = int(numpy.math.ceil(N/len(final_clusters)))

            for indices in final_clusters:
                sub_matrix = matrix[indices]
                new_sub_matrix = samples_generation(sub_matrix, batch_count)
                new_negatives = new_negatives + list(new_sub_matrix)
                generated_rownames = list("neg_"+str(i+j) for i in range(batch_count-len(indices)))
                if len(generated_rownames) == 0:
                    continue
                j = j + len(generated_rownames)
                rownames = rownames + generated_rownames        
            return rownames, new_negatives        
        
        numpy.random.seed(1234)
        N = 200
        features = self.features["positives"][:]
        if len(self.examples["negatives"]) == 0:
            rownames, new_negatives = generate_artificials(N, 0)
        else:
            N1 = int(numpy.ceil(0.75*N))
            N2 = int(numpy.ceil(0.25*N))           
            rownames_1, new_negatives_1 = generate_artificials(N1, len(self.examples["negatives"]))
            rownames_2, new_negatives_2 = clone_negatives(N2, N1+len(self.examples["negatives"]))
            rownames = rownames_1 + rownames_2
            new_negatives = new_negatives_1 + new_negatives_2
            #rownames, new_negatives = clone_negatives(N, 0)


    
        for i in range(len(rownames)):
            rowname = rownames[i]
            try:
                self.examples_pool[rowname]
            except KeyError:
                for j in range(len(features)):
                    feature = features[j]
                    self.features_pool.setdefault(feature, []).append(rowname)
                    self.examples_pool.setdefault(rowname, {}).update({feature:new_negatives[i][j]})
        
        self.examples["final_negatives"] = rownames[:]




    def get_neg_features(self):
        tmp_client, tmp_db, tmp_col = self.connect_to_mongo(self.SP.db, self.mongo_res_col)
        results = tmp_col.find({'query_id':{'$in':self.examples["final_negatives"]}})
        tmp_neg_features = []
        for entry in results:
            tmp_neg_features.append(entry["domain_id"])
            self.features_pool.setdefault(entry["domain_id"], []).append(entry["query_id"])
            domain = entry["domain_id"]
            max_score = round(float(self.max_scores_dict[domain]),1)
            abs_score = float(entry["score"])
            normalized_score = 100*round(abs_score/max_score, 4)
            self.examples_pool.setdefault(entry["query_id"], 
                                          {}).update({domain : normalized_score})
        
        tmp_neg_features = list(set(tmp_neg_features))[:]
        for feature in self.features["positives"]:
            try:
                tmp_neg_features.remove(feature)
            except ValueError:
                pass
        self.features["negatives"] = tmp_neg_features[:]
        F = open(self.file_name, 'a')
        F.write("features_from_negatives,"+str(len(self.features["negatives"]))+"\n")
        F.close()


    def add_neg_features(self):
        denominator = float(len(self.examples["negatives"]))
        tmp_list = []
        for feature in self.features["negatives"]:
            ratio = len(self.features_pool[feature])/denominator
            tmp_list.append([feature,ratio])
        tmp_tuples = sorted(tmp_list, key=operator.itemgetter(1), reverse=True)    
        
        denominator = float(len(self.features["positives"]))
        new_features = []
        for tmp_tuple in tmp_tuples:
            feature =  tmp_tuple[0]
            score = tmp_tuple[1]
            if score < 0.5:
                break
            else:  
                new_features.append(feature)
                if (len(new_features)/denominator > 0.2 or len(new_features)==10):
                    break
                else:
                    pass
        return new_features        

    def construct_framework(self, limit=True):
        tmp_list = self.get_pos_examples()
        if len(tmp_list) == 0:
            F = open(self.file_name, 'a')
            F.write("Process termination,No positive examples\n")
            F.close()
            return 1
        tmp_list = self.get_pos_features()
        if len(tmp_list) == 2:
            print self.ec_number
        if len(tmp_list) == 0:
            F = open(self.file_name, 'a')
            F.write("Process termination,No positive features\n")
            F.close()
            return 1
        tmp_list = self.get_neg_examples()
        
        if len(tmp_list) < 150: # threshold is 200, give a little space
            self.generate_negatives()
            F = open(self.file_name, 'a')
            F.write("generate_negatives,True\n")
        else:
            self.get_upper_classes_neg_examples()
            self.finalize_negatives_population()
            self.get_neg_features()
            additional_features = self.add_neg_features()   
            for feature in additional_features:
                self.features.setdefault("final_negatives", []).append(feature)  
        F = open(self.file_name, 'a')
        F.write("features_from_negatives_approved,"+ 
        str(len(self.features["final_negatives"]))+"\n")
        F.close()

        

        ### GLOBAL COMMANDS
        self.examples_list = self.examples["positives"] + self.examples["final_negatives"]
        features = list(set(self.features["positives"] + self.features["final_negatives"]))
        self.features_list = sorted(features)        
        for example in self.examples["final_negatives"]:
            self.classes.update({example:self.negative_class})
        for example in self.examples["positives"]:
            self.classes.update({example:self.positive_class})  
        return 0

    def set_training_data(self):
        column_names = ["class"] + self.features_list
        rows_data = []
        for tmp_example, tmp_class in self.classes.items():
            tmp_data = []
            tmp_data.append(tmp_class)
            for tmp_feature in self.features_list:
                try:
                    tmp_data.append(self.examples_pool[tmp_example][tmp_feature])
                except:
                    tmp_data.append(0)
            rows_data.append(tmp_data)
            
        self.df = pandas.DataFrame(numpy.array(rows_data), columns=column_names,
                                   index=self.classes.keys())
        #print self.df
        return self.df
            
            
    def deal_with_imbalance(self):

        negatives_df = self.df[self.df['class'] != self.ec_number]
        neg_count = float(negatives_df.shape[0])
        positives_df = self.df[self.df['class'] == self.ec_number]
        pos_count = float(positives_df.shape[0])
        ratio = round(numpy.log2(neg_count) - numpy.log2(pos_count),2)
        F = open(self.file_name, 'a')
        F.write("initial_populations_log_ratio,"+str(ratio)+"\n")
        F.write("initial_populations_natural_ratio,"+str(round(numpy.power(2,ratio),2))+"\n")
        
        if ratio > 0.32:
            # attempt 1: remove duplicates
            tmp_df = negatives_df.ix[:, negatives_df.columns != 'class']
            nr_tmp_df = remove_duplicates_from_df(tmp_df)
            negatives_filtered = list(nr_tmp_df.index)
            positives = list(positives_df.index)
            rownames = negatives_filtered + positives
            self.df = self.df.ix[rownames]
            self.examples["final_negatives"] = negatives_filtered[:]
            ratio = round(numpy.log2(len(negatives_filtered)) - numpy.log2(pos_count),2)
            F.write("negatives_without_duplicates,"+str(len(negatives_filtered))+"\n")
            F.write("updated_populations_log_ratio,"+str(ratio)+"\n")
            if ratio > 0.32:
                self.df = class_imbalance_oversampling(self.df, 'class', 'positive_class')
                F.write("oversampling,"+"positive_class"+"\n")
                positives_df = self.df[self.df['class'] == self.ec_number]
                negatives_df = self.df[self.df['class'] != self.ec_number]
                ids = list(i for i in positives_df.index if i.startswith("pos"))
                ratio = round(numpy.log2(negatives_df.shape[0]) - numpy.log2(positives_df.shape[0]),2)
                F.write("generated_examples,"+str(len(ids))+"\n")
                F.write("updated_populations_log_ratio,"+str(ratio)+"\n")
        elif ratio < -0.32:
            self.df = class_imbalance_oversampling(self.df, 'class', 'negative_class')
            F.write("oversampling,"+"negative_class"+"\n")
            positives_df = self.df[self.df['class'] == self.ec_number]
            negatives_df = self.df[self.df['class'] != self.ec_number]
            ids = list(i for i in negatives_df.index if i.startswith("neg"))
            ratio = round(numpy.log2(negatives_df.shape[0]) - numpy.log2(positives_df.shape[0]),2)
            F.write("generated_examples,"+str(len(ids))+"\n")
            F.write("updated_populations_log_ratio,"+str(ratio)+"\n")
        else:
            pass

        F.close()
                
 
    def summarization(self):        
        numpy.random.seed(1234)
        colors = {"no_"+self.ec_number:"#323030", "seed_no_"+self.ec_number:"#848282", 
                  self.ec_number:"#882426", "seed_"+self.ec_number:"#C39192"}
        positives_df = self.df[self.df['class'] == self.ec_number]
        negatives_df = self.df[self.df['class'] != self.ec_number]

        tmp_dict = {1: self.ec_number, 2:"no_"+self.ec_number}
        min_value = 0
        max_value = 100
        linspace = numpy.linspace(min_value, max_value, num=500)
        
        for feature in self.features["positives"]:

            plt.figure(figsize=(20,12), dpi=300)
            max_freqs = []
            for key in sorted(tmp_dict.keys()):
                if key == 1:
                    tmp_df = positives_df
                else:
                    tmp_df = negatives_df
                values =  list(tmp_df[feature].astype(float))
                if len(values) == 1:
                    values.append(values[0]+numpy.random.uniform(0,1))
                    values.append(values[0]-numpy.random.uniform(0,1))
                elif len(values) == 0:
                    continue
                else:
                    pass
                try:
                    kde = gaussian_kde(values)
                    kde.set_bandwidth(bw_method=kde.factor / 1.5)             
                    y = kde(linspace)
                    max_freqs.append(max(kde.pdf(linspace)))
                except numpy.linalg.linalg.LinAlgError:
                    y =  [0]*len(linspace)

                plt.plot(linspace, y, label=tmp_dict[key],
                         c=colors[tmp_dict[key]], linewidth=3.0)
                         
                plt.fill_between(linspace, y, color=colors[tmp_dict[key]], 
                                 alpha='0.20')
            max_freq = max(max_freqs)
            plt.grid(color='grey', linestyle='-')
            plt.xlabel(feature+" bit score", size=24, labelpad=24)
            plt.xlim([min_value, max_value+2])
            plt.xticks(range(10,110,10), size=16)
            plt.ylabel("Frequency", size=24, labelpad=24)
            plt.ylim([0,max_freq+0.02])
            plt.xlim([0,100])
            plt.yticks([])
            plt.title(feature+ " bit score distributions", size=32, y=1.02)
            plt.legend(loc='upper right', ncol=1, borderaxespad=1, fontsize=24)
            # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.savefig
            plt.savefig(self.directory+feature+".png")        
            plt.close('all')

    def write_training_data(self):
        self.df.to_csv(self.directory+self.ec_number+".csv")
