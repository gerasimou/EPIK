import pandas as pd

data = [34.999871253789586, 8.45411921652242, 1.7711272962424118]
headers = ['v1', 'v2', 'P1']

dataLoL = list(map(lambda e1: [e1], data))
df = pd.DataFrame(data=dataLoL, columns=headers)

