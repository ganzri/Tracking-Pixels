#plot figure 5.3 of thesis. 

import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [107431 , 1793, 9075]

y_p = [12.2, 1.5, 7.3]

plt.bar(x, y_p, tick_label=["heuristic", "model", "filter lists"])

plt.xlabel("Method")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Percentage Mismatches")
plt.savefig('./bars.pdf', bbox_inches='tight')
