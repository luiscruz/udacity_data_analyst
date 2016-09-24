#!/usr/bin/python

import sys
import signal
import pickle
import numpy as np
import itertools
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score,precision_score
sys.path.append("./tools/")

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

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list_original = ['poi','salary'] # You will need to use more features
features_list_with_pois = features_list_original + ['from_this_person_to_poi','from_poi_to_this_person']
features_list_with_expenses = features_list_original + ['expenses']
features_all = [
    'salary',
    'to_messages',
    'deferral_payments',
    'total_payments',
    'exercised_stock_options',
    'bonus',
    'restricted_stock',
    'shared_receipt_with_poi',
    'restricted_stock_deferred',
    'total_stock_value',
    'expenses',
    'loan_advances',
    'from_messages',
    'other',
    'from_this_person_to_poi',
    'director_fees',
    'deferred_income',
    'long_term_incentive',
    # 'email_address',
    'from_poi_to_this_person',
]
features_list=features_all

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
best_solution_so_far = None
optimization_path = []

#cycle to try several features
for L in range(2, 5):
    for features_list in itertools.permutations(features_all, L):
        try:
            print ""
            print "------------------------------------------------------------"
            print ""
            print " New iteration with following features "
            print features_list
            data = featureFormat(my_dataset, features_list, sort_keys = True)
            labels, features = targetFeatureSplit(data)
    
    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html
    
    #  In previous experiments GaussianNB performed well. For the sake 
    # of simplicity we will be using it
            classifiers = [
                GaussianNB(),
                # RandomForestClassifier(n_estimators=100),
            ]
    
    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    
    # Example starting point. Try investigating other evaluation techniques!        
            features_train, features_test, labels_train, labels_test = \
                train_test_split(features, labels, test_size=0.3, random_state=42)
            
            # function that evaluates each classifier
            def evaluate_classifier(classifier):
                classifier.fit(features_train, labels_train)
                predictions = classifier.predict(features_test)
                return (
                    precision_score(labels_test, predictions, pos_label=1, average='macro'),
                    recall_score(labels_test, predictions, pos_label=1, average='macro'),
                    f1_score(labels_test, predictions, pos_label=1, average='macro'),
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