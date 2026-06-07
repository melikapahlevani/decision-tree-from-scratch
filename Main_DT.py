# ============================================================================
# LIBRARY IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import unittest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# DECISION TREE CLASSES IMPORT
# ============================================================================

from DT_Library import Node
from DT_Library import DecisionTree