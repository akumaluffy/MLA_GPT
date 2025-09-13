import matplotlib.pyplot as plt
import numpy as np

models = ("10L:8H", "10L:10H", "10L:12H", "12L:8H", "12L:10H", "12L:12H", "14L:8H", "14L:10H", "14L:12H")

avg_loss = {
    "MLA": (0.242, 0.2848, 0.2895, 0.243, 0.2794, 0.2899, 0.2318, 1.4832, 0.2788),
    "Flash": (3.7580, 3.7626, 3.7729, 3.7129, 3.7682, 3.7088, 3.7039, 3.7251, 3.7283)
    }


perplexity = {
    "MLA": (1.27, 1.33, 1.34, 1.28, 1.32, 1.34, 1.26, 4.41, 1.32),
    "Flash" : (42.86, 43.06, 43.50, 40.97, 43.30, 40.80, 40.61, 41.47, 41.61)
}

best_loss = {
    "MLA": (0.1006, 0.1259, 0.1336, 0.1006, 0.1336, 0.1488, 0.1045, 1.2397, 0.1432),
    "Flash": (3.2411, 3.2592, 3.2373, 3.2259, 3.2522, 3.2215, 3.2239, 3.2268, 3.2166)
}

bar_colors = ['darkgreen', 'goldenrod']
x = np.arange(len(models))
width = 0.2
multiplier = 0

 # Set global font size for tick labels
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

fig1, axs1 = plt.subplots(1, 1, figsize = (6, 3.92), layout='constrained')
cat = 0
for architecture, value in avg_loss.items():
    offset = width * multiplier
    rects1 = axs1.bar(x + offset, value, width, label=architecture, color=bar_colors[cat])
    axs1.bar_label(rects1, padding=0.8, fontsize=8)
    multiplier+=1
    cat = (-1) * cat + 1

axs1.set_ylabel("Average Loss Value", fontsize=8)
axs1.set_xlabel("Model Configuration", fontsize=8)
axs1.set_xticks(x + width, models, fontsize=8, rotation=90)
axs1.set_title(f"Average Test Dataset Loss Values", fontsize =10)
#axs1.legend(ncols = 2)
axs1.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig1.tight_layout()

#plt.show()

fig2, axs2 = plt.subplots(1, 1, figsize = (6, 3.92), layout='constrained')
multiplier = 0
cat = 0
for architecture, value in perplexity.items():
    offset = width * multiplier
    rects2 = axs2.bar(x + offset, value, width, label=architecture, color=bar_colors[cat])
    axs2.bar_label(rects2, padding=0.8, fontsize=8)
    multiplier += 1
    cat = (-1) * cat + 1

axs2.set_ylabel("Perplexity Value", fontsize=8)
axs2.set_xlabel("Model Configuration", fontsize=8)
axs2.set_xticks(x + width, models, fontsize=8, rotation=90)
axs2.set_title(f"Test Datastet Perplexity Values", fontsize=10)
#axs2.legend(ncols = 2)
axs2.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig2.tight_layout()

fig3, axs3 = plt.subplots(1, 1, figsize = (6, 3.92), layout='constrained')
multiplier = 0
cat = 0
for architecture, value in best_loss.items():
    offset = width * multiplier
    rects3 = axs3.bar(x + offset, value, width, label=architecture, color=bar_colors[cat])
    axs3.bar_label(rects3, padding=0.8, fontsize=8)
    multiplier += 1
    cat = (-1) * cat + 1


axs3.set_ylabel("Loss Value")
axs3.set_xlabel("Model Configuration",fontsize=8)
axs3.set_xticks(x + width, models, fontsize=8,rotation=90)
axs3.set_title(f"Best Model Loss Values",fontsize=10)
#axs3.legend(ncols = 2)
axs3.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig3.tight_layout()

plt.show()