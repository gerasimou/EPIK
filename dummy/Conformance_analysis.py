

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


DATA_DIR = '../data'
CONFORMANCE_OPTIONS = [0, 1, 2, 3]
conformance_files = ['FPR_Conformance_' + str(i) + '_BestSolutions.csv' for i in CONFORMANCE_OPTIONS]
conformanceLabels = ['Conformance', 'Low Conflict', 'Medium Conflict', 'High Conflict']
colHeaders = [conformanceLabels[i] for i in CONFORMANCE_OPTIONS]

from EPIK_visualisation import do_conformance_boxplots

do_conformance_boxplots(data_dir=DATA_DIR, conformance_files=conformance_files, colHeaders=colHeaders)


# from EPIK_visualisation import prepare_conformance_data
# dfData = prepare_conformance_data(DATA_DIR, conformance_files, colHeaders)

# dfData.head()