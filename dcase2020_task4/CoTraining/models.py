import sys
import os

dirname = os.path.dirname(os.path.abspath( __file__ ))
one_before = os.path.join("/", *str(dirname).split("/")[:-1])

sys.path.append(one_before)
print(sys.path)

# Import the models from the baseline and the the last year system
from dcase2019.models import *
from baseline.models import *
