from classifier import build_classifier
from hypothesis_tests import hypo_test_on_numerical
import sys

def main(tragets):
    if 'build-classifier' in targets:
        build_classifier("system_sysinfo_unique_normalized.csv000")

    if 'hypo-test' in targets:
        hypo_test_on_numerical("system_sysinfo_unique_normalized.csv000",'ram')

    if 'test' in targets:
        build_classifier("system_sysinfo_unique_normalized.csv000")

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
