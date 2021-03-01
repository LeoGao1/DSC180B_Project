
import sys
import json


sys.path.insert(0, 'src/classifier')
sys.path.insert(0, 'src/hypothesis_test')
sys.path.insert(0, 'src/etl')

from classifier import build_classifier
from hypothesis_tests import hypo_test_on_numerical
from clean import data_cleaning

data_path = "cleaned_data/system_sysinfo_unique_normalized.csv000"
cleaned_data_path = "cleaned_data/"
all_classifier_list = ['knn','random forest',
                        'decision tree','neural network',
                        'svm','sgd','logistic']

result_path = 'result/classifier_performance.csv'

def main(tragets):
    if 'clean_data' in targets:
        data_cleaning(data_path,cleaned_data_path)

    if 'build-classifier' in targets:
        data_cleaning(data_path,cleaned_data_path)
        build_classifier(data_path)

    if 'hypo-test' in targets:
        hypo_test_on_numerical(data_path,'ram')

    if 'test' in targets:
        data_cleaning(data_path,cleaned_data_path)
        build_classifier(data_path)

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
