"""This old combined test has been split into three standalone scripts.

Run these commands from the project root:

1. Direct hard-threshold backward error:
   conda run -n torch python test\01_direct_hard_threshold_error.py

2. One-batch gradient probe:
   conda run -n torch python test\02_no_surrogate_gradient_probe.py

3. Short training comparison:
   conda run -n torch python test\03_no_surrogate_training_compare.py
"""


def main():
    print(__doc__)


if __name__ == '__main__':
    main()