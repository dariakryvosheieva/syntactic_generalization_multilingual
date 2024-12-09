import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import binomtest
from pandas import DataFrame

from results import Index, Cols, scores

total_trials = 1000
expected_probability = 0.5

def one_tailed_test(successes, alternative):
    return binomtest(successes, total_trials, expected_probability, alternative=alternative).pvalue

significance_scores = []
for i in range(len(Index)):
    model_significance_scores = []
    for j in range(len(Cols)):
        successes = int(scores[i][j] * total_trials)
        p_above_chance = one_tailed_test(successes, 'greater')
        p_below_chance = one_tailed_test(successes, 'less')
        p = min(p_above_chance, p_below_chance)
        if p == p_above_chance and p < 0.05:
            model_significance_scores.append(1)
        elif p == p_below_chance and p < 0.05:
            model_significance_scores.append(-1)
        else:
            model_significance_scores.append(0)
    significance_scores.append(model_significance_scores)

df = DataFrame(significance_scores, index=Index, columns=Cols)

plt.figure(figsize=(20, 20))

ax = sns.heatmap(df, cmap='RdYlGn', square=True, cbar=False)
ax.hlines(list(range(len(Index) + 1)), *ax.get_xlim(), color="black", lw=0.5)
ax.vlines(list(range(len(Cols) + 1)), *ax.get_ylim(), color="black", lw=0.5)

plt.xticks(rotation=70)
plt.yticks(rotation=0)

plt.savefig('significance.pdf')
