import sys
import csv
import pandas as pd

csv_input = sys.argv[1]
csv_output = sys.argv[2]
df = pd.read_csv(csv_input, encoding='utf-8', delimiter=",")
classes = len(df)
dfs = []
for i in range(classes):
    for j in range(classes):
        cur = pd.DataFrame([{"row":i + 1, "col":j + 1, "value":df.ix[i][j]}])
        dfs.append(cur)
result = pd.concat(dfs, ignore_index=True)
result.to_csv(csv_output, sep=',', index=False, header=["row", "col", "value"])
