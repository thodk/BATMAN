# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 19:46:33 2017

@author: thodoris
"""
import numpy
import pandas
import operator
import copy
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score



def pca_selection(X, n_components=0.95, pca_coefs_threshold=0.01):
    
    pca = PCA(n_components=n_components)
    pca.fit(X)
    #print "number of components:", pca.n_components_
    #print "top 10 PC, explained_variance:", pca.explained_variance_ratio_[0:5]*100
    components = list(pca.components_)
    features_dict = {}
    for j in range(len(components)):
        variance = pca.explained_variance_ratio_[j]
        coefs = zip(range(X.shape[1]), list(pca.components_[j]))
        coefs = list( tuple([i[0], abs(i[1])] for i in coefs))
        coefs = sorted(coefs, key=operator.itemgetter(1), reverse=True)
        for tup in coefs:
            if tup[1] > pca_coefs_threshold:
                features_dict.setdefault(tup[0], []).append(tup[1]*variance)
            else:
                break

    features_list = []
    for feature, array_of_scores in features_dict.items():
        score = sum(array_of_scores)
        features_list.append([feature, score])
    
    features_list = sorted(features_list, key=operator.itemgetter(1), reverse=True)
    features_list = list(i[0] for i in features_list)
    #print "initial features:", X.shape[1] 
    #print "final features:", len(features_list)

    return features_list


def cv_adaboost(X, Y, ranked_features_list, pos_label, recursion_step=0.05, 
                cv=10, max_n_features=400):        
    if recursion_step < 1.:
        batch = int(len(ranked_features_list)*float(recursion_step))
    else:
        batch = recursion_step

    if X.shape[1] > 1000:
        max_n_features = int(X.shape[1]*0.5)
    else:
        pass

    base_estimator = DecisionTreeClassifier(max_depth=6)
    estimator = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=10, random_state=1234)
    batches_scores = {}
    i=0

    for n_features in range(batch, len(ranked_features_list), batch):
        if n_features > max_n_features:
            break
        i+=1
        tmp_features = ranked_features_list[:n_features][:]
        tmp_X = X[:,tmp_features]

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1234)
        splits = skf.split(X=tmp_X, y=Y)
        sensitivity = []
        #specificity = []
        for train, test in splits:
            train_x = tmp_X[train]; train_y = Y[train]
            test_x = tmp_X[test]; test_y = Y[test]
            estimator.fit(train_x, train_y)
            predictions = estimator.predict(test_x)
            sensitivity.append(recall_score(y_pred=predictions, y_true=test_y, pos_label=pos_label))
            #specificity.append(recall_score(y_pred=predictions, y_true=test_y, pos_label='no_'+pos_label))

        score = numpy.mean(sensitivity)
        
        #print n_features, score, numpy.mean(specificity)
        batches_scores.update({i:{"score":score, "n_features":n_features}})

    return batches_scores



def check_monotone(values, monotone_dec_threshold=0.02, monotone_inc_threshold=0.005):
    intervals_monotone = {}
    for i in range(len(values)):
        if i == 0:
            current = values[i]
        else:
            current = values[i]
            previous = values[i-1]
            diff = current - previous
            if diff > 0 and diff >= monotone_inc_threshold:
                monotone = "increasing function"
            elif diff > 0 and diff < monotone_inc_threshold:
                monotone = "weak increasing function"
            elif diff < 0 and abs(diff) >= monotone_dec_threshold:
                monotone = "decreasing function"
            else:
                monotone = "weak decreasing function"
            intervals_monotone.update({i:monotone})
    return intervals_monotone


def find_crucial_interval(intervals_monotone):
    valid_intervals = []
    for i in range(len(intervals_monotone), 0, -1):
        monotone = intervals_monotone[i]
        if monotone != 'decreasing function':
            valid_intervals.append(i)
        else:
            if len(valid_intervals) == 0:
                continue
            else:
                break
    return sorted(valid_intervals)


def calculate_ss_tot(values):
    mean_value = numpy.mean(values)
    ss_tot = sum(numpy.power((values - mean_value), 2))
    return ss_tot

def calculate_ss_res(true, pred):
    ss_res = sum(numpy.power((pred - true), 2))
    return ss_res

def calculate_r_squared(true, pred):
    ss_tot = calculate_ss_tot(true)
    ss_res = calculate_ss_res(true, pred)
    r_s = 1 - (ss_res / float(ss_tot))
    return r_s
    


def find_maximum(X, Y, batches_scores):

    y_values = list(batches_scores[i]["score"] for i in batches_scores.keys())
    x_values = list(batches_scores[i]["n_features"] for i in batches_scores.keys())
    intervals_monotone = check_monotone(y_values)
    valid_intervals = find_crucial_interval(intervals_monotone)

    new_y_values = numpy.array(y_values)[valid_intervals]
    new_x_values = numpy.array(x_values)[valid_intervals]

    X_dict = {}
    X_dict.update({1:new_x_values})
    logs = numpy.array(list(numpy.log(v) for v in new_x_values))
    X_dict.update({2:logs})
    for i in range(2,6):
        powers = numpy.array(list(v**i for v in new_x_values))
        X_dict.update({i+1:powers})    
    
    X = pandas.DataFrame(X_dict).as_matrix()
    Y = numpy.array(new_y_values).reshape(-1,1)

    for i in range(1, 7):
        LR = LinearRegression()
        LR.fit(X[:,0:i], Y)
        true = Y
        pred = LR.predict(X[:,0:i])
        '''
        from matplotlib import pyplot as plt
        plt.figure(numpy.random.choice(100))
        plt.scatter(list(X[:,0]), true)
        plt.plot(list(X[:,0]), true)
        plt.scatter(list(X[:,0]), pred, color='r')
        plt.plot(list(X[:,0]), pred, c='r')
        plt.ylim([0.8,1])
        plt.show()
        '''
        r_squared = calculate_r_squared(true, pred)

        if r_squared >= 0.99:
            model = copy.deepcopy(LR)
            break
        else:
            model = copy.deepcopy(LR)

    coefficients = list(model.coef_[0])
    derivative_coefs = []
    derivative_coefs.append(coefficients[0])
    derivative_coefs.append(coefficients[1])
    for i in range(2,len(coefficients)):
        derivative_coefs.append(coefficients[i]*i)
    
    
    derivatives = []
    tmp_x = []; tmp_y = []
    for x in range(min(new_x_values), max(new_x_values)+1, 1):
        y = 0
        y = y + derivative_coefs[0]
        y = y + derivative_coefs[1]*1/float(x)
        for j in range(2,len(derivative_coefs)):
            y = y + derivative_coefs[j]*numpy.power(x,j-1)
        tmp_x.append(x); tmp_y.append(y)
        if y >= 0:
            derivatives.append([x, y])
    '''
    from matplotlib import pyplot as plt
    plt.figure(numpy.random.choice(100))
    plt.plot(tmp_x, tmp_y)
    plt.ylim([min(tmp_y),max(tmp_y)])
    plt.show()
    '''    
    best_x_value = sorted(derivatives, key=operator.itemgetter(1))[0][0]
    
    return best_x_value




def exe(data_frame, pos_label, n_components=0.95, pca_coefs_threshold=0.01,
        recursion_step=0.05, cv=10, max_n_features=400, 
        monotone_dec_threshold=0.02, monotone_inc_threshold=0.005):

    df_X = data_frame.ix[:,data_frame.columns != 'class']
    X = df_X.as_matrix()
    
    df_Y = data_frame.ix[:, data_frame.columns == 'class']
    Y = df_Y['class'].as_matrix()
    
    ranked_features = pca_selection(X)
    scores = cv_adaboost(X, Y, ranked_features, pos_label)
    best_value = find_maximum(X, Y, scores)
    
    best_features_indices = ranked_features[:best_value]
    best_features = list(df_X.columns.values[best_features_indices])
    best_features.insert(0, 'class')
    return data_frame[list(best_features)]
    
    
