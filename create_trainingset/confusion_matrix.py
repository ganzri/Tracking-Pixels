"""
Could be used to plot confusion matrices, need to copy in the actual matrices, there is an automated script however
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from matplotlib import gridspec

#confusion  matrix between ground-truth from consent declarations and prediction by the model
matrix = [[11495.6,   566.4,   686. ,   243.4],
       [  163.4,  1585.8,   167.2,    98. ],
       [  260.2,   441.8, 14575. ,   401.8],
       [  241.4,  1566. ,  1872.8, 13876.4]]

for i in range(4):
    row_tot = sum(matrix[i])
    for j in range(4):
        matrix[i][j] /= row_tot

labels = np.array([[f'{v:.2f}'.lstrip('0')[:3] for v in row] for row in matrix])

df_cm = pd.DataFrame(matrix, index=[i for i in ('Necessary', 'Functional', 'Analytical', 'Advertisement')],
                     columns=[i for i in ('Necessary', 'Functional', 'Analytical', 'Advertisement')])

plt.figure(figsize=(5,3))
sn.heatmap(data=df_cm, annot_kws={'fontsize': 12}, annot=labels, fmt='', cmap='YlGnBu', cbar=True, vmin=0, vmax=1)
plt.xlabel('Prediction')
plt.ylabel('Ground truth')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0, ha='right')

plt.savefig('./GT_PRED_conf_matrix.pdf', bbox_inches='tight')

#confusion matrix between ground-truth from consent notice and decision of the filter lists
matrix_2=[[12, 13],
        [34, 56],
        [34, 67],
        [98, 56]]

for i in range(4):
    row_tot = sum(matrix_2[i])
    for j in range(2):
        matrix_2[i][j] /= row_tot

labels = np.array([[f'{v:.2f}'.lstrip('0')[:3] for v in row] for row in matrix_2])

df_cm = pd.DataFrame(matrix_2, index=[i for i in ('Necessary', 'Functional', 'Analytical', 'Advertisement')],
                     columns=[i for i in ('not blocked', 'blocked')])



plt.figure(figsize=(3,3))
sn.heatmap(data=df_cm, annot_kws={'fontsize': 12}, annot=labels, fmt='', cmap='YlGnBu', cbar=True, vmin=0, vmax=1)
plt.xlabel('Filter List')
plt.ylabel('Ground truth')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0, ha='right')

plt.savefig('./GT_FL_conf_matrix.pdf', bbox_inches='tight')

#confusion matrix between prediction of the model and decision of the filter list
matrix_3=[[12, 13],
        [34, 56],
        [34, 67],
        [98, 56]]

for i in range(4):
    row_tot = sum(matrix_3[i])
    for j in range(2):
        matrix_3[i][j] /= row_tot

labels = np.array([[f'{v:.2f}'.lstrip('0')[:3] for v in row] for row in matrix_3])

df_cm = pd.DataFrame(matrix_3, index=[i for i in ('Necessary', 'Functional', 'Analytical', 'Advertisement')],
                     columns=[i for i in ('not blocked', 'blocked')])



plt.figure(figsize=(3,3))
sn.heatmap(data=df_cm, annot_kws={'fontsize': 12}, annot=labels, fmt='', cmap='YlGnBu', cbar=True, vmin=0, vmax=1)
plt.xlabel('Filter List')
plt.ylabel('Ground truth')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0, ha='right')

plt.savefig('./PRED_FL_conf_matrix.pdf', bbox_inches='tight')

