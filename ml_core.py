# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 19:20:29 2017

@author: thodoris
"""

import pandas
import numpy
import os
import copy
import core


from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


class Performance(object):
    
    def __init__(self, y_true, y_pred, pos_probas, pos_label, neg_label):
        self.accuracy = round(metrics.accuracy_score(y_true, y_pred), 4)        
        self.sensitivity = round(metrics.recall_score(y_true, y_pred, pos_label=pos_label), 4)
        self.specificity = round(metrics.recall_score(y_true, y_pred, pos_label=neg_label), 4)
        self.mcc = round(metrics.matthews_corrcoef(y_true, y_pred), 4)
        
        fpr, tpr, thresholds = metrics.roc_curve(y_true, pos_probas, pos_label=pos_label)
        self.auc = round(metrics.auc(fpr, tpr), 4)

    def summary(self):
        record = [self.accuracy, self.sensitivity, self.specificity,
                  self.mcc, self.auc]
        return record


class AveragePerformance(object):

    def __init__(self, pos_label, neg_label):
        self.accuracy_list = []
        self.sensitivity_list = []
        self.specificity_list = []
        self.mcc_list = []
        self.auc_list = []
        self.confusion_matrix = []
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.update()

    def mean_worker(self, tmp_list):
        if tmp_list == []:
            return 0
        else:
            m = round(numpy.mean(tmp_list), 4)
            return m
    
    def std_worker(self, tmp_list):
        if tmp_list == []:
            return 0
        else:
            s = round(numpy.std(tmp_list), 4)
            return s
    
    def update(self):
        self.accuracy = self.mean_worker(self.accuracy_list)
        self.accuracy_std = self.std_worker(self.accuracy_list)       
        self.sensitivity = self.mean_worker(self.sensitivity_list)
        self.sensitivity_std = self.std_worker(self.sensitivity_list)       
        self.specificity = self.mean_worker(self.specificity_list)
        self.specificity_std = self.std_worker(self.specificity_list)
        self.mcc = self.mean_worker(self.mcc_list)
        self.mcc_std = self.std_worker(self.mcc_list)        
        self.auc = self.mean_worker(self.auc_list)
        self.auc_std = self.std_worker(self.auc_list)
        #self.confusion_matrix
    
    def append(self, y_true, y_pred, pos_probas):
        tmp = Performance(y_true, y_pred, pos_probas, self.pos_label, self.neg_label)
        self.accuracy_list.append(tmp.accuracy)
        self.sensitivity_list.append(tmp.sensitivity)
        self.specificity_list.append(tmp.specificity)
        self.mcc_list.append(tmp.mcc)
        self.auc_list.append(tmp.auc)
        self.update()
        
    def summary(self):
        record = [self.accuracy, self.accuracy_std, 
                  self.sensitivity, self.sensitivity_std,
                  self.specificity, self.specificity_std,
                  self.mcc, self.mcc_std,
                  self.auc, self.auc_std]
        return record





class ModelValidation(object):

    def __init__(self, classifier, pos_label, neg_label):
        self.classifier = classifier
        self.pos_label = pos_label
        self.neg_label = neg_label

    def index_of_pos_class(self, classes, pos_class):
        return list(classes).index(pos_class)
   
    def cross_validation(self, learning_X, learning_y, n_splits=10):
	if learning_X.shape[0] < 30:
	    n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)
        splits = skf.split(learning_X, learning_y)
        performance = AveragePerformance(self.pos_label, self.neg_label)
        for train, test in splits:
            self.classifier.fit(learning_X[train], learning_y[train]) 
            probas = self.classifier.predict_proba(learning_X[test])
            predictions = self.classifier.predict(learning_X[test])
            class_index = self.index_of_pos_class(self.classifier.classes_, 
                                                  self.pos_label)
            pos_probas = list(entry[class_index] for entry in list(probas))
            performance.append(learning_y[test], predictions, pos_probas)
        results = performance.summary()
        return results

    def validation(self, learning_X, learning_y, validation_X, validation_y):
        self.classifier.fit(learning_X, learning_y)
        probas = self.classifier.predict_proba(validation_X)
        predictions = self.classifier.predict(validation_X)
        class_index = self.index_of_pos_class(self.classifier.classes_, self.pos_label)
        pos_probas = list(entry[class_index] for entry in list(probas))
        performance = Performance(validation_y, predictions, pos_probas, 
                                  self.pos_label, self.neg_label)
        results = performance.summary()
        return results






class GridSearchParam(object):
    
    def __init__(self):
        self.parameters = []
        self.param_combinations = []
        self.name = None
        self.classifier = None
        self.results = {}

    def record_results(self, process, input_list):
        if process == "learning":
            self.results.setdefault("learning", []).append(input_list)
        elif process == "validation":
            self.results.setdefault("validation", []).append(input_list)
        else:
            pass

    def parameterization(self, x, y, pos_label, neg_label, cv=10, 
                         validation_size=0.3):
        numpy.random.seed(1234)
        splitted_data = train_test_split(x, y, test_size=validation_size, stratify=y)
        learning_X = splitted_data[0]
        learning_y = splitted_data[2].ravel()
        validation_X = splitted_data[1]
        validation_y = splitted_data[3].ravel()

        if self.param_combinations == []:
            param_id = self.name + "_None"
            model = ModelValidation(self.classifier, pos_label, neg_label) 
            cv_results = model.cross_validation(learning_X, learning_y)
            valid_results = model.validation(learning_X, learning_y,
                                             validation_X, validation_y)
            cv_results.insert(0, {})
            cv_results.insert(0, self.name)
            cv_results.insert(0, param_id)
            self.record_results("learning", cv_results)
            valid_results.insert(0, {})
            valid_results.insert(0, self.name)
            valid_results.insert(0, param_id)           
            self.record_results("validation", valid_results)                                            
        else:
            for param_set in self.param_combinations:
                print self.name, param_set
                param_id = "_".join([str(i) for i in param_set.values()])
                param_id = self.name + "_" +param_id
                self.classifier.set_parameters(param_set)
                model = ModelValidation(self.classifier, pos_label, neg_label) 
                cv_results = model.cross_validation(learning_X, learning_y)
                valid_results = model.validation(learning_X, learning_y,
                                                 validation_X, validation_y)
                
                cv_results.insert(0, param_set)
                cv_results.insert(0, self.name)
                cv_results.insert(0, param_id)
                self.record_results("learning", cv_results)
                valid_results.insert(0, param_set)
                valid_results.insert(0, self.name)
                valid_results.insert(0, param_id)            
                self.record_results("validation", valid_results)
        

    def get_cv_results(self):           
        columns = ["ID", "Classifier" ,"Parameters", "Accuracy", "Accuracy_std", 
        "Sensitivity", "Sensitivity_std", "Specificity", "Specificity_std", 
        "MCC", "MCC_std","AUC", "AUC_std"]
        df = pandas.DataFrame(self.results["learning"], columns=columns)
        return df

    def get_validation_results(self):           
        columns = ["ID", "Classifier" ,"Parameters", "Accuracy", "Sensitivity", 
        "Specificity", "MCC", "AUC"]
        df = pandas.DataFrame(self.results["validation"], columns=columns)
        return df





class ChangeParameters(object):

    def set_parameters(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)






class NaiveBayesGridSearchParam(GridSearchParam):
    
    def __init__(self):
        self.name = "gaussian_naive_bayes"
        self.parameters = []
        self.classifier = NaiveBayes()
        self.results = {}

    def get_parameters(self):
        self.param_combinations = []        
        return self.param_combinations

class NaiveBayes(GaussianNB, ChangeParameters):
    
    pass




class LogisticRegressionGridSearchParam(GridSearchParam):
    '''
    > The loss function to be used. Defaults to ‘hinge’, which gives a linear SVM. 
    The ‘log’ loss gives logistic regression, a probabilistic classifier. 
    ‘modified_huber’ is another smooth loss that brings tolerance to outliers 
    as well as probability estimates. ‘squared_hinge’ is like hinge but 
    is quadratically penalized. ‘perceptron’ is the linear loss used by 
    the perceptron algorithm.
    
    > Using loss="log" or loss="modified_huber" enables the predict_proba
    
    > Penalties:
    penalty="l2": L2 norm penalty on coef_
    penalty="l1": L1 norm penalty on coef_
    penalty="elasticnet": Convex combination of L2 and L1; (1-l1_ratio)*L2+l1_ratio*L1
    '''
    
    def __init__(self, dimensionality):
        self.name = "logistic_regression"
        self.dimensionality = dimensionality
        self.classifier = LogisticRegression()
        self.results = {}
        if self.dimensionality <= 5:
            l1_ratios = [0.]
        elif self.dimensionality <= 10:
            l1_ratios = [0. , 0.2]
        elif self.dimensionality <=20:
            l1_ratios = [0., 0.2, 0.4, 0.6]
        else:
            l1_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        self.parameters = [{'loss':['log', 'modified_huber'], 
                           'penalty':['elasticnet'],
                           'l1_ratio':l1_ratios, 'alpha':alphas, 
                           'class_weight':['balanced', None]}]

    def get_parameters(self):
        self.param_combinations = []
        for comb in ParameterGrid(self.parameters):
            self.param_combinations.append(comb)                     
        return self.param_combinations

class LogisticRegression(SGDClassifier, ChangeParameters):
    pass




class DecisionTree(DecisionTreeClassifier, ChangeParameters):
    pass

class DecisionTreeGridSearchParam(GridSearchParam):

    def __init__(self, dimensionality):
        self.name = "decision_tree"
        self.classifier = DecisionTree()
        self.dimensionality = dimensionality
        self.results = {}
        
        highest_depth = 20
        if self.dimensionality < 3:
            valid_depths = [self.dimensionality]
        elif self.dimensionality < 5:
            valid_depths = [2, self.dimensionality]            
        else:
            valid_depths = [2] + range(4, min([highest_depth, self.dimensionality]), 2)

        self.parameters = [{'criterion':['gini'], 
                           'max_depth':valid_depths, 
                           'max_features':[0.6, 1.0], 
                           'class_weight':['balanced', None]}]
    
        self.param_combinations = []
        for comb in ParameterGrid(self.parameters):
            self.param_combinations.append(comb)

    def get_parameters(self):
        return self.param_combinations




class SVM(SVC, ChangeParameters):
    pass


class SVMGridSearchParam(GridSearchParam):
    '''
    http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#linear
    '''
    def __init__(self, dimensionality):
        self.name = "support_vector_machines"
        self.classifier = SVM(probability=True)
        self.results = {}
        if dimensionality == 1:
            self.parameters = [
            {"C":[0.1, 1.], 'kernel':['linear', 'rbf'], 'class_weight':['balanced',None]},
            {"C":[0.1, 1.], 'kernel':['sigmoid'], 'gamma':['auto', 0.001, 0.01, 0.1],
             'class_weight':['balanced', None]}
             ]
        else:   
            self.parameters = [
            {"C":[0.1, 1., 10.], 'kernel':['linear', 'rbf'], 'class_weight':['balanced',None]},
            {"C":[0.1, 1.], 'kernel':['poly'], 'degree':[2], 'class_weight':['balanced',None]},
            {"C":[0.1, 1.], 'kernel':['sigmoid'], 'gamma':['auto', 0.001, 0.01, 0.1],
             'class_weight':['balanced', None]}
             ]

    def get_parameters(self):

        self.param_combinations = []
        for comb in ParameterGrid(self.parameters):
            self.param_combinations.append(comb)
        return self.param_combinations



def max_estimators(input_x):
    x = numpy.linspace(-100, 100, 200)  
    max_old = 100.
    max_new = 300.
    min_old = -100.
    min_new = 0.    
    x_scaled = (x-min_old)*(max_new-min_new)/(max_old-min_old) + min_new
    x_discr = numpy.ceil(numpy.ceil(x_scaled)/10.)*10.

    y = 1/(1+numpy.exp(-0.04*x))    
    max_old = 1.
    max_new = 300.
    min_old = 0.
    min_new = 1.       
    y_scaled = (y-min_old)*(max_new-min_new)/(max_old-min_old) + min_new
    y_discr = numpy.ceil(numpy.ceil(y_scaled)/10.)*10.
    
    tuples = zip(x_discr, y_discr)
    func = {}
    for tup in tuples:
        key = tup[0]
        value = tup[1]
        func.setdefault(key, []).append(value)
    
    for key, values in func.items():
        func[key] = max(values)

    
    input_x_discr = numpy.ceil(numpy.ceil(input_x)/10.)*10.
    return int(func[input_x_discr])



class RandomForest(RandomForestClassifier, ChangeParameters):
    pass


class RandomForestGridSearchParam(GridSearchParam):
    
    def __init__(self, dimensionality):
        self.name = "random_forest"
        self.classifier = RandomForest()
        self.dimensionality = dimensionality
        self.results = {}
        
        DTGSP = DecisionTreeGridSearchParam(dimensionality)
        tree_parameters = DTGSP.parameters

        if dimensionality <= 5:
            n_estimators = [2, 4]
            valid_depths  = [2,3]
        elif dimensionality <= 10:
            n_estimators = [2, 5, 10]
            valid_depths  = [2,3]            
        else:
            if dimensionality >= 300:
                max_value = 300.
            else:
                max_value = max_estimators(dimensionality)
            n_estimators = range(0, max_value+1, 20)
            n_estimators.remove(0)
            valid_depths  = [2,3,4]

        self.parameters = []

        for entry in tree_parameters:
            entry.update({'n_estimators':n_estimators})
            entry.update({'max_depth':valid_depths})
            self.parameters.append(entry)

    def get_parameters(self):
        self.param_combinations = []
        for comb in ParameterGrid(self.parameters):
            self.param_combinations.append(comb)        
        return self.param_combinations            


class AdaBoost(AdaBoostClassifier, ChangeParameters):
    pass

class AdaBoostGridSearchParam(GridSearchParam):
    
    def __init__(self, dimensionality):
        self.name = "adaboost"
        self.classifier = AdaBoost()
        self.dimensionality = dimensionality
        self.results = {}
        
        DTGSP = DecisionTreeGridSearchParam(dimensionality)
        tree_parameters = DTGSP.parameters

        if dimensionality <= 5:
            n_estimators = [2, 4]
            valid_depths  = [2,3]
        elif dimensionality <= 10:
            n_estimators = [2, 5, 10]
            valid_depths  = [2,3]            
        else:
            if dimensionality >= 300:
                max_value = 300.
            else:
                max_value = max_estimators(dimensionality)
            n_estimators = range(0, max_value+1, 20)
            n_estimators.remove(0)
            valid_depths  = [2,3,4]

        self.parameters = []

        for entry in tree_parameters:
            entry.update({'n_estimators':n_estimators})
            entry.update({'max_depth':valid_depths})
            self.parameters.append(entry)

    def get_parameters(self):
        self.param_combinations = []
        for comb in ParameterGrid(self.parameters):
            self.param_combinations.append(comb)        
        return self.param_combinations



class TrainingDatathon(object):

    def __init__(self, ec_number, mongo_db_name, hmm_db_name, main_dir):
        CF = core.ClassifierFramework(ec_number, hmm_db_name, mongo_db_name, 
        "data", "ec_numbers", "training_"+hmm_db_name, main_dir)
        integer = CF.construct_framework()
        if integer == 0:    
            CF.set_training_data()
            CF.deal_with_imbalance()
            CF.summarization()
            CF.write_training_data()



class Classificathon(object):

    def __init__(self, ec_number, main_dir, data_frame=None):
        ec_numbers_linkage = core.find_ec_number_linkage(ec_number)
        path = "/".join(ec_numbers_linkage)+"/"
        self.whole_dir = main_dir + path
        if os.path.isfile(self.whole_dir+ec_number+".csv"):
            self.csv = self.whole_dir+ec_number+".csv"
            self.df = pandas.read_csv(self.csv, index_col=0)
            self.y = self.df['class'].as_matrix().ravel()
            self.x = self.df.ix[:,self.df.columns != 'class'].as_matrix()
            self.pos_label = ec_number
            self.neg_label = "no_"+ec_number
            self.learning_scores_file = self.whole_dir+"cv_results.csv"
            self.validation_scores_file = self.whole_dir+"valid_results.csv"
        else:
            print "no training data"
            return
        
        
    def construct_classifiers(self):

        dimensionality = self.x.shape[1]
        tmp = ["gaussian_naive_bayes", "logistic_regression", "decision_tree",
        "support_vector_machines", "random_forest", "adaboost"]

        for method in tmp:
            if method == "gaussian_naive_bayes":
                GSP = NaiveBayesGridSearchParam()
                GSP.get_parameters()
                GSP.parameterization(self.x, self.y, self.pos_label, self.neg_label)
                nv_cv_results = GSP.get_cv_results()
                nv_valid_results = GSP.get_validation_results()                  
            elif method == "logistic_regression":
                GSP = LogisticRegressionGridSearchParam(dimensionality)
                GSP.get_parameters()
                GSP.parameterization(self.x, self.y, self.pos_label, self.neg_label)
                log_reg_cv_results = GSP.get_cv_results()
                log_reg_valid_results = GSP.get_validation_results()  
            elif method == "decision_tree":
                GSP = DecisionTreeGridSearchParam(dimensionality)
                GSP.get_parameters()
                GSP.parameterization(self.x, self.y, self.pos_label, self.neg_label)
                dec_tree_cv_results = GSP.get_cv_results()
                dec_tree_valid_results = GSP.get_validation_results()                
            elif method == "support_vector_machines":
                GSP = SVMGridSearchParam(dimensionality)
                GSP.get_parameters()                
                GSP.parameterization(self.x, self.y, self.pos_label, self.neg_label)
                svm_cv_results = GSP.get_cv_results()
                svm_valid_results = GSP.get_validation_results()  
            elif method == "random_forest":
                GSP = RandomForestGridSearchParam(dimensionality)
                GSP.get_parameters()
                GSP.parameterization(self.x, self.y, self.pos_label, self.neg_label)
                rf_cv_results = GSP.get_cv_results()
                rf_valid_results = GSP.get_validation_results()
            elif method == "adaboost":
                GSP = AdaBoostGridSearchParam(dimensionality)
                GSP.get_parameters()
                GSP.parameterization(self.x, self.y, self.pos_label, self.neg_label)
                ada_cv_results = GSP.get_cv_results()
                ada_valid_results = GSP.get_validation_results()
            else:
                pass

        
        valid_results_df = pandas.concat([nv_valid_results,
                                          log_reg_valid_results, 
                                          dec_tree_valid_results,
                                          svm_valid_results,
                                          rf_valid_results,
                                          ada_valid_results],
                                          ignore_index=True)
        valid_results_df = valid_results_df.sort_values("MCC", ascending=False)
        self.valid_results_df = valid_results_df                              
        valid_results_df.to_csv(self.validation_scores_file)  

                                
        cv_results_df = pandas.concat([nv_cv_results,
                                       log_reg_cv_results,
                                       dec_tree_cv_results,
                                       svm_cv_results,
                                       rf_cv_results,
                                       ada_cv_results],
                                       ignore_index=True)                                  
        cv_results_df = cv_results_df.sort_values("MCC", ascending=False)
        self.cv_results_df = cv_results_df
        cv_results_df.to_csv(self.learning_scores_file)


    def get_best_estimator_params(self, top=20):
        
        cv_top = self.cv_results_df.ix[:top][["ID", "MCC", "Parameters"]]
        valid_top = self.valid_results_df.ix[:top][["ID", "MCC", "Parameters"]]
        print cv_top.shape
        tmp_dict = {}
        for i in range(top):
            print cv_top.ix[i,:]
            tmp_id = cv_top.ix[i,:]["ID"]
            cv_score = cv_top.ix[i,:]["MCC"]
            param = cv_top.ix[i,:]["Parameters"]
            tmp_dict.setfefault(tmp_id, {}).update({"parameters":param})
            tmp_dict.setfefault(tmp_id, {}).setdefault("scores", []).append(0.4*cv_score)
            
            tmp_id = valid_top.ix[i,:]["ID"]
            valid_score = valid_top.ix[i,:]["MCC"]
            param = valid_top.ix[i,:]["Parameters"]
            tmp_dict.setfefault(tmp_id, {}).update({"parameters":param})
            tmp_dict.setfefault(tmp_id, {}).setdefault("scores", []).append(0.6*valid_score)

        for key, values in tmp_dict.values():
            final_score = round(sum(values["scores"]), 4)
            values.update({"final_score":final_score})
            tmp_dict.update({key:values})

        
        '''        
        best_params = ranked_parameters[0]
        best_estimator = self.estimator
        for key, value in best_params.items():
            setattr(best_estimator, key, value)
        best_estimator.fit(X, y)
        return best_estimator
        '''
    def construct_best_estimator(self):
        pass











'''     
#p = P('3.1.4.4')    
#p = P('3.5.1.23')
#p = P('3.1.1.1')
#p = P('4.1.99.19')

p = P('1.14.13.165')
#p = P('3.4.22.10')
ecs = [
'6.2.1.36',
'2.7.1.72',
'4.1.1.21',
'3.2.1.93',
'4.1.99.19',
'4.1.1.44',
'2.4.1.251',
'3.5.2.2',
'1.3.8.1',
'3.5.1.24',
'3.5.1.23',
'2.4.1.175',
'1.11.1.18',
'1.7.7.-',
'1.7.1.6',
'1.7.1.7',
'1.7.1.4',
'2.4.1.230',
'6.3.5.9',
'3.4.14.12',
'1.1.1.309',
'3.4.14.11',
'1.14.13.114',
'1.14.13.113',
'1.7.7.1',
'1.7.7.2',
'1.5.8.2',
'2.1.1.218',
'2.1.1.219',
'2.1.1.210',
'2.1.1.213',
'2.1.1.215',
'2.1.1.216',
'2.1.1.217',
'3.5.4.3',
'3.7.1.4',
'3.5.4.1',
'1.2.1.-',
'1.2.3.3',
'1.14.12.14',
'1.14.12.15',
'1.14.12.17',
'1.14.12.10',
'1.14.12.11',
'1.14.12.12',
'1.2.1.8',
'1.2.1.7',
'1.2.1.5',
'1.2.1.4',
'1.2.1.3',
'1.14.12.19',
'1.4.3.3',
'4.2.1.19',
]

#for ec in ecs:
    #p = P(ec)
'''
