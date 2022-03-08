#calculate mean and stdev for some measures from 5-fold CV
from statistics import mean, stdev

##change to values reported by CV:
#overall measures:
tot_acc_ratio = [ 0.9879974420778198, 0.9873764080869694, 0.9874010034925476, 0.987702372827163, 0.9881512362189715]
macro_recall = [ 0.943856051842705, 0.8930688373155623, 0.9934205844889771, 0.9934544845711573, 0.9686155605076301]

#per class measures
prec_1 = [0.99878932, 0.8, 0.97591947, 0.94570679]
prec_2 = [0.99837217, 0.81818182, 0.9767967, 0.94355044]
prec_3 = [0.99876844, 0.75, 0.97610599, 0.94172093]
prec_4 = [0.99870856, 0.8, 0.97722008, 0.94346665]
prec_5 = [0.99856001, 0.9, 0.97743191, 0.94700544]

recall_1 = [0.98606871, 0.8, 0.99604868, 0.99330682]
recall_2 = [0.98575282, 0.6, 0.99484536, 0.99167717]
recall_3 = [0.98541885, 1., 0.99501608, 0.99324741]
recall_4 = [0.98582413, 1.,0.99481173, 0.99318208]
recall_5 = [0.98647753, 0.9, 0.99540339, 0.99258132] 

##calculate and print statistics
classes = ["Necessary", "Functional", "Analytics", "Advertising"]

print(f"Total accuracy ratio (mean, stdev): {mean(tot_acc_ratio)}, {stdev(tot_acc_ratio)}")
print(f"Macro recall (mean, stdev): {mean(macro_recall)}, {stdev(macro_recall)}")

for i in range(4):
    prec = [prec_1[i], prec_2[i], prec_3[i], prec_4[i], prec_5[i]]
    recall = [recall_1[i], recall_2[i], recall_3[i], recall_4[i], recall_5[i]]
    print(f"{classes[i]}: Precision (mean, stdev): {mean(prec)}, {stdev(prec)}")
    print(f"{classes[i]}: Recall (mean, stdev): {mean(recall)}, {stdev(recall)}")

