import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

font = FontProperties()
font.set_family("monospace")

blobs_DB = [(1.11,0.51), (1.09,0.46), (0.96,0.48), (0.85,0.38), (0.85,0.38)]
blobs_DB_2 = [(1.16,0.43), (1.06,0.48), (1.02,0.42), (0.83,0.41), (0.67,0.28)]

class_DB = [(1.87,0.58),(1.89,0.59),(1.92,0.63),(1.98,0.64),(1.98,0.75)]
class_DB_2 = [(1.89,0.55),(1.89,0.60),(1.93,0.59),(1.94,0.69),(1.98,0.70)]

insects_DB = [(2.73,0.39),(2.63,0.41),(2.09,0.43),(2.65,0.25)]
insects_DB_2 = [(2.63,0.31),(2.61,0.40),(2.39,0.41),(2.17,0.25)]

covert_DB = [(0.19,0.01),(0.16,0.01),(0.16,0.04),(0.09,0.02)]
covert_DB_2 = [(0.17,0.02),(0.15,0.01),(0.11,0.02),(0.09,0.03)]

blobs_AUC = [(0.73,0.11),(0.8,0.09),(0.86,0.08),(0.92,0.06),(0.96,0.03)]
blobs_AUC_2 = [(0.81,0.10),(0.8,0.11),(0.83,0.08),(0.91,0.07),(0.96,0.04)]

class_AUC = [(0.59,0.1),(0.58,0.07),(0.61,0.11),(0.61,0.08),(0.63,0.1)]
class_AUC_2 = [(0.57,0.07),(0.58,0.08),(0.59,0.08),(0.63,0.11),(0.63,0.09)]

insects_AUC = [(0.57,0.04),(0.55,0.05),(0.64,0.04),(0.52,0.03)]
insects_AUC_2 = [(0.50,0.06),(0.50,0.04),(0.60,0.05),(0.63,0.05)]

covert_AUC = [(0.93,0.01),(0.54,0.02),(0.87,0.02),(0.6,0.03)]
covert_AUC_2 = [(0.59,0.04),(0.81,0.04),(0.59,0.02),(0.54,0.02)]

betas_syn = [0.1,0.25,0.4,0.6,0.75]
betas_real = [0.1,0.3,0.5,0.7]

pearson_data_AUC = ([b[0] for b in blobs_AUC] + [b[0] for b in blobs_AUC_2] + [b[0] for b in class_AUC] +
                    [b[0] for b in class_AUC_2] + [b[0] for b in insects_AUC] + [b[0] for b in insects_AUC_2] +
                    [b[0] for b in covert_AUC] + [b[0] for b in covert_AUC_2])

pearson_data_DB = ([b[0] for b in blobs_DB] + [b[0] for b in blobs_DB_2] + [b[0] for b in class_DB] +
                   [b[0] for b in class_DB_2] + [b[0] for b in insects_DB] + [b[0] for b in insects_DB_2] +
                   [b[0] for b in covert_DB] + [b[0] for b in covert_DB_2])

corr, _ = pearsonr(pearson_data_AUC, pearson_data_DB)

plt.style.use('seaborn-v0_8-colorblind')

# ax = plt.subplot(1,1,1)
# ax.plot(betas_syn,[b[0] for b in blobs_DB])
# ax.plot(betas_syn,[b[0] for b in class_DB])
# ax.plot(betas_real,[b[0] for b in insects_DB])

ax = plt.subplot(1,1,1)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.scatter([b[0] for b in blobs_AUC],[b[0] for b in blobs_DB],s=80,label='isoGauss', color='#0173b2', linewidth=1)
ax.scatter([b[0] for b in blobs_AUC_2],[b[0] for b in blobs_DB_2],s=80, color='#0173b2')
ax.scatter([b[0] for b in class_AUC],[b[0] for b in class_DB],s=80,label='hyperCube', color='#029e73')
ax.scatter([b[0] for b in class_AUC_2],[b[0] for b in class_DB_2],s=80, color='#029e73')
ax.scatter([b[0] for b in insects_AUC],[b[0] for b in insects_DB],s=80,label='insects', color='#d55e00')
ax.scatter([b[0] for b in insects_AUC_2],[b[0] for b in insects_DB_2],s=80, color='#d55e00')
ax.scatter([b[0] for b in covert_AUC],[b[0] for b in covert_DB],s=80,label='covertype', color='#de8f05')
ax.scatter([b[0] for b in covert_AUC_2],[b[0] for b in covert_DB_2],s=80, color='#de8f05')
plt.plot([], [], ' ', label="PearsonR:"+str(round(corr,2)))

L = ax.legend(facecolor='white', framealpha=1,fontsize=16)
plt.setp(L.texts, family='monospace')
ax.tick_params(axis='x', colors='gray')
ax.tick_params(axis='y', colors='gray')
ax.spines['top'].set_color('white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
ax.set_xlabel('Average AUC',fontsize=16, color='gray')
ax.set_ylabel('Average D-B index',fontsize=16, color='gray')
ax.set_facecolor("#e0e0e0")
ax.grid(color='white')
ax.set_axisbelow(True)

plt.gcf().set_size_inches(8, 5.5)
plt.show()

# plt.xlabel("That is verbatim", fontproperties=font)
# plt.ylabel("That is normal")


