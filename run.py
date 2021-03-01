
import sys
import json


sys.path.insert(0, 'src/classifier')
sys.path.insert(0, 'src/hypothesis_test')
#sys.path.insert(0, 'src/model')

from classifier import build_classifier
from hypothesis_tests import hypo_test_on_numerical


data_path = "cleaned_data/system_sysinfo_unique_normalized.csv000"

def main(tragets):
    if 'build-classifier' in targets:
        build_classifier(data_path)

    if 'hypo-test' in targets:
        hypo_test_on_numerical(data_path,'ram')

    if 'test' in targets:
        build_classifier(data_path)

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
