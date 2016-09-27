#!/usr/bin/python

import sys
import signal
import pickle
import numpy as np
import itertools
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score,recall_score,precision_score
sys.path.append("./tools/")
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

def finish():
    try:
        if best_solution_so_far:

            print ""
            print "============================="
            print "Optimization Path:"
            print "============================="
            for solution in optimization_path:
                print_result_item(solution[0])
                print solution[1]
            print ""
            print "============================="
            print "Final Solution:"
            print "============================="
            print best_solution_so_far
            clf = best_solution_so_far[0][3]
            features_list = best_solution_so_far[1]
            dump_classifier_and_data(clf, my_dataset, features_list)
            print "Model saved with success."
        else:
            print ""
            print "No solution found"
    except Exception as e:
        print e

def signal_handler(signal, frame):
    finish()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

### Task 1: Select what primary features you'll use.
features_top10 = [
 'exercised_stock_options',
 'total_stock_value',
 'bonus',
 'salary',
 'salary_bonus_ratio',
 'deferred_income',
 'long_term_incentive',
 'restricted_stock',
 'total_payments',
 'shared_receipt_with_poi',
 ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

del data_dict['TOTAL']

### Task 3: Create new feature(s)

#add salary_bonus_ratio = bonus/salary
for _,obj in data_dict.items():
    salary_bonus_ratio = np.float(obj['bonus'])/np.float(obj['salary'])
    if np.isnan(salary_bonus_ratio):
        salary_bonus_ratio = -1
    obj['salary_bonus_ratio'] = salary_bonus_ratio

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
best_solution_so_far = None
optimization_path = []
#cycle to try several features
for L in range(2, 5):
    # for features in [features_tail5[:]]:
    for features in itertools.permutations(features_top10, L):
        features_list = ['poi'] + list(features)
        try:
            print ""
            print "------------------------------------------------------------"
            print ""
            print " New iteration with following features "
            print features_list
            data = featureFormat(my_dataset, features_list, sort_keys = True)
            labels, features = targetFeatureSplit(data)
            labels = np.array(labels)
            features = np.array(features)
    
    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html
    
    #  In previous experiments GaussianNB performed well. For the sake 
    # of simplicity we will be using it
            classifiers = [
                GaussianNB(),
                QuadraticDiscriminantAnalysis(),
            ]
    
    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    
    # Example starting point. Try investigating other evaluation techniques!        
            sss = StratifiedShuffleSplit(labels, 3, test_size=0.3, random_state=42)
            # function that evaluates each classifier
            def evaluate_classifier(classifier):
                global_predictions = []
                global_labels = []
                for train_index, test_index in sss:
                    features_train, features_test = features[train_index], features[test_index]
                    labels_train, labels_test = labels[train_index], labels[test_index]
                    classifier.fit(features_train, labels_train)
                    predictions = classifier.predict(features_test)
                    global_labels = np.append(global_labels, labels_test)
                    global_predictions = np.append(global_predictions,predictions)
                return (
                    precision_score(global_labels, global_predictions, pos_label=1, average='binary'),
                    recall_score(global_labels, global_predictions, pos_label=1, average='binary'),
                    f1_score(global_labels, global_predictions, pos_label=1, average='binary'),
                    classifier
                )
                    
            # evaluate every classifier and select the one with highest f1score
            results = [evaluate_classifier(classifier) for classifier in classifiers]
            print "--------------------------------------------------"
            print "%25s   %4s   %4s   %4s"%("Classifier name","P", "R", "F1")
            print "--------------------------------------------------"
            def print_result_item(result_item):
                (precision, recall, f1score, clf) = result_item
                clf_name = clf.__class__.__name__
                print "%25s   %.2f   %.2f   %.2f"%(clf_name,precision, recall, f1score)
            for result_item in results:
                print_result_item(result_item)
            print "--------------------------------------------------"
            
            best_model_index = np.argmax(zip(*results)[1])
            best_model_in_this_iteration = results[best_model_index]
            precision,max_recall,f1score, clf = best_model_in_this_iteration
            ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
            print 'Selected %s model in this iteration:'%(ordinal(best_model_index+1))
            print_result_item(best_model_in_this_iteration)
            if best_solution_so_far:
                print "Best solution"
                print_result_item(best_solution_so_far[0])
            if best_solution_so_far != None:
                (precision_so_far,recall_so_far, f1score_so_far, _) = best_solution_so_far[0]
            optimization_function = lambda p,r,f1 : p+1.2*r-abs(p-r)    
            
            if best_solution_so_far == None or optimization_function(precision_so_far,recall_so_far, f1score_so_far) <= optimization_function(precision,max_recall,f1score) :
                best_solution_so_far = (best_model_in_this_iteration, features_list)
                optimization_path.append(best_solution_so_far)
        except Exception as e:
            print e
    print ""
    print "============================="
    print "    Best Solution so far"
    print "============================="
    print best_solution_so_far

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

finish()