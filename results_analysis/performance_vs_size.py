import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from results import *

mgpt_params = [1417596928, 13108070400]
mgpt_scores = scores[1:3]

bloom_params = [559214592, 1065314304, 1722408960, 3002557440, 7069016064, 176247271424]
bloom_scores = scores[3:9]

xglm_params = [564463616, 1732907008, 2941505536, 7492771840]
xglm_scores = scores[9:12]
xglm_scores.append(scores[13])

xlmr_params = [278295186, 560142482, 3482741760, 10712994816]
xlmr_scores = scores[15:]

def get_suite_scores(model_scores, suite_id):
    return [model_scores[i][suite_id] for i in range(len(model_scores))]

fig, axs = plt.subplots(10, 2, figsize=(11, 30))

for suite_id in range(len(Cols)):
    subplot_i, subplot_j = suite_id // 2, suite_id % 2
    line_1 = axs[subplot_i, subplot_j].plot(mgpt_params, get_suite_scores(mgpt_scores, suite_id), 'o-', color="orange", label='mGPT')
    line_2 = axs[subplot_i, subplot_j].plot(bloom_params, get_suite_scores(bloom_scores, suite_id), 'o-', color="blue", label='BLOOM')
    line_3 = axs[subplot_i, subplot_j].plot(xglm_params, get_suite_scores(xglm_scores, suite_id), 'o-', color="green", label='XGLM')
    line_4 = axs[subplot_i, subplot_j].plot(xlmr_params, get_suite_scores(xlmr_scores, suite_id), 'o-', color="red", label='XLM-R')
    axs[subplot_i, subplot_j].set_xscale('log')
    axs[subplot_i, subplot_j].set_title(Cols[suite_id], fontsize=14)
    axs[subplot_i, subplot_j].set_xlabel('Number of parameters', fontsize=14)
    axs[subplot_i, subplot_j].set_ylabel('Accuracy', fontsize=14)
    axs[subplot_i, subplot_j].tick_params(axis='both', which='major', labelsize=14)

plt.subplots_adjust(left=0.1, right=0.9, hspace=1, wspace=0.3) 

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.09, 0.945), fontsize=14)

fig.suptitle('Accuracy versus number of parameters', x=0.295, y=0.96, fontsize=16)

# plt.savefig('performance_vs_size.pdf')

slopes = [[], [], [], []]
for suite_id in range(len(Cols)):
    slopes[0].append(np.polyfit(np.array(mgpt_params) / 1e9, np.array(get_suite_scores(mgpt_scores, suite_id)) * 100, 1)[0])
    slopes[1].append(np.polyfit(np.array(bloom_params) / 1e9, np.array(get_suite_scores(bloom_scores, suite_id)) * 100, 1)[0])
    slopes[2].append(np.polyfit(np.array(xglm_params) / 1e9, np.array(get_suite_scores(xglm_scores, suite_id)) * 100, 1)[0])
    slopes[3].append(np.polyfit(np.array(xlmr_params) / 1e9, np.array(get_suite_scores(xlmr_scores, suite_id)) * 100, 1)[0])
df = DataFrame(slopes, index=['mGPT', 'BLOOM', 'XGLM', 'XLM-R'], columns=Cols)
df.to_csv('performance_vs_size.csv')

