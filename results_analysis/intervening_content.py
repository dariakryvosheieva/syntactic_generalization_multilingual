import matplotlib.pyplot as plt
from results import *

models = ['mGPT', 'BLOOM', 'XGLM', 'mBERT', 'XLM-R']

df['model'] = ['human'] + ['mGPT' for _ in range(2)] + ['BLOOM' for _ in range(6)] + ['XGLM' for _ in range(5)] + ['mBERT'] + ['XLM-R' for _ in range(4)]
df_mean = df.groupby('model').mean()
df_sem = df.groupby('model').sem()

fig, axs = plt.subplots(4, 1, figsize=(12, 20))

suite_classes = [
    ['hindi-S_ne_O_V', 'hindi-S_ne_PossPRN_O_V', 'hindi-S_ne_PossPRN_PossN_O_V'],
    ['hindi-S_O_V', 'hindi-S_PossPRN_O_V', 'hindi-S_PossPRN_PossN_O_V'],
    ['swahili-N_of_Poss_V', 'swahili-N_of_Poss_D_V', 'swahili-N_of_Poss_D_A_V'],
    ['swahili-N_of_Poss_D_ni_A', 'swahili-N_of_Poss_D_AP_ni_AN', 'swahili-N_of_Poss_D_AP_V_ni_AN']
]

for subplot_i in range(4):
    suites = suite_classes[subplot_i]
    for model in models:
        line = axs[subplot_i].errorbar(suites, [df_mean.loc[model, suite] for suite in suites], yerr=[df_sem.loc[model, suite] for suite in suites], fmt='o-', label=model)
    axs[subplot_i].set_xlabel('Test suite', fontsize=16)
    axs[subplot_i].set_ylabel('Accuracy', fontsize=16)
    axs[subplot_i].tick_params(axis='both', which='major', labelsize=14)
    axs[subplot_i].margins(x=0.2)

axs[0].set_title('Hindi (ergative-absolutive)', fontsize=16)
axs[1].set_title('Hindi (nominative-accusative)', fontsize=16)
axs[2].set_title('Swahili (verbal predicate)', fontsize=16)
axs[3].set_title('Swahili (adjectival predicate)', fontsize=16)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.115, 0.98), fontsize=14)

plt.suptitle('Accuracy versus complexity of intervening phrase', x=0.375, y=1, fontsize=18)

plt.subplots_adjust(hspace=0.4)

plt.savefig('intervening_content.pdf')

avg = 0
for model in Index[1:]:
    avg += df.loc[model, 'swahili-N_of_Poss_D_V'] - df.loc[model, 'swahili-N_of_Poss_V']
print(avg / 18)