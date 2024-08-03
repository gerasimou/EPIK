

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



DATA_DIR = 'data'
CONFORMANCE_OPTIONS = [0, 1, 2, 3]
filenames = ['FPR_Conformance_C' + str(i) + '_KL.csv' for i in CONFORMANCE_OPTIONS]
# conformanceLabels = ['Conformance', 'Low Conflict', 'Medium Conflict', 'High Conflict']



GENERATIONS = 50

# colHeaders = ['C'+str(CONFORMANCE_OPTIONS[i]) for i in CONFORMANCE_OPTIONS]
colHeaders = ['Conformance', 'Light conflict', 'Medium conflict', 'Heavy conflict']
dfData = pd.DataFrame(columns=colHeaders)

# KL_traces = []

for i in range(len(filenames)):
    filename = filenames[i]
    filepath = "../"+ os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath, header=None)

    KL_trace = []
    for gen in range(GENERATIONS):
        rows = df.shape[0]+1
        population = int(rows/GENERATIONS)

        dfI = df.iloc[gen*population:(gen+1)*population, :]
        KL_trace.append(np.median(np.sum(dfI.T)))

    # KL_traces.append(KL_trace)
    dfData.iloc[:, i] = KL_trace


removedEntries = 3
dfDataMelt = pd.melt(dfData.iloc[removedEntries:,])
dfDataMelt['index'] = np.tile(np.arange(GENERATIONS-removedEntries)+1, 4)


ax = sns.lineplot(style='variable', hue='variable', y='value', x='index', data=dfDataMelt)
ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=14)
ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=14)

ax.set_xticks([0, 10, 20, 30, 40, 48], labels=['0', '10', '20', '30', '40', '50'])
ax.set_xlabel("Generation", fontsize=14)
ax.set_ylabel("Distance (median per generation)", fontsize=14)
ax.legend(title=None)




plotName = "KL_Evolution_median.pdf"
plt.savefig(plotName, bbox_inches='tight', pad_inches=0.02)

plt.show()

