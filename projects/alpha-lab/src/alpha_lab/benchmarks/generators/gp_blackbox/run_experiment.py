import sys
import os
from strategy import strategy  # import experiment's strategy

sys.path.insert(0, os.environ['ALPHALAB_WORKSPACE'])
from harness import blackbox  # import the blackbox function

blackbox.evaluate(strategy)  # run the experiment