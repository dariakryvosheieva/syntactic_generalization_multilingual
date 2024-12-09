import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

from pandas import DataFrame

Index = [
    'human',
    'mGPT-1.3B',
    'mGPT-13B',
    'bloom-560m',
    'bloom-1b1',
    'bloom-1b7',
    'bloom-3b',
    'bloom-7b1',
    'bloom',
    'xglm-564M',
    'xglm-1.7B',
    'xglm-2.9B',
    'xglm-4.5B',
    'xglm-7.5B',
    'mbert',
    'xlmr-base',
    'xlmr-large',
    'xlmr-xl',
    'xlmr-xxl',
]

Cols = [
    'basque-DO-S_DO_V_AUX',
    'basque-DO-S_IO_DO_V_AUX',
    'basque-IO-IO_S_V_AUX',
    'basque-IO-S_IO_DO_V_AUX',
    'basque-S-IO_S_V_AUX',
    'basque-S-S_DO_V_AUX',
    'basque-S-S_IO_DO_V_AUX',
    'basque-S-S_V_AUX',
    'hindi-S_ne_O_V',
    'hindi-S_ne_PossPRN_O_V',
    'hindi-S_ne_PossPRN_PossN_O_V',
    'hindi-S_O_V',
    'hindi-S_PossPRN_O_V',
    'hindi-S_PossPRN_PossN_O_V',
    'swahili-N_of_Poss_D_A_V',
    'swahili-N_of_Poss_D_AP_ni_AN',
    'swahili-N_of_Poss_D_AP_V_ni_AN',
    'swahili-N_of_Poss_D_ni_A',
    'swahili-N_of_Poss_D_V',
    'swahili-N_of_Poss_V'
]

scores = [
    [1, 0.92, 0.98, 0.82, 0.86, 0.7, 0.92, 0.86, 0.84, 0.92, 0.82, 0.82, 0.88, 0.74, 0.74, 0.78, 0.64, 0.66, 0.78, 0.86],
    [0.996, 0.978, 0.91, 0.973, 0.991, 0.897, 0.75, 0.913, 0.956, 0.966, 0.974, 0.831, 0.914, 0.933, 0.408, 0.564, 0.527, 0.594, 0.444, 0.668],
    [0.991, 0.988, 0.794, 0.994, 0.999, 0.861, 0.916, 0.836, 0.95, 0.929, 0.944, 0.805, 0.922, 0.938, 0.481, 0.594, 0.542, 0.61, 0.494, 0.709],
    [0.868, 0.145, 0.123, 0.531, 0.323, 0.687, 0.519, 0.744, 0.875, 0.783, 0.868, 0.841, 0.855, 0.886, 0.444, 0.482, 0.457, 0.502, 0.378, 0.487],
    [0.945, 0.534, 0.093, 0.794, 0.168, 0.692, 0.669, 0.827, 0.882, 0.916, 0.918, 0.818, 0.867, 0.897, 0.41, 0.514, 0.502, 0.527, 0.341, 0.444],
    [0.918, 0.644, 0.326, 0.773, 0.656, 0.688, 0.643, 0.712, 0.902, 0.903, 0.924, 0.859, 0.889, 0.898, 0.395, 0.452, 0.468, 0.498, 0.345, 0.494],
    [0.981, 0.754, 0.452, 0.836, 0.753, 0.853, 0.773, 0.882, 0.858, 0.913, 0.917, 0.829, 0.896, 0.904, 0.378, 0.481, 0.463, 0.539, 0.339, 0.494],
    [0.981, 0.92, 0.724, 0.932, 0.928, 0.869, 0.864, 0.801, 0.869, 0.904, 0.921, 0.839, 0.901, 0.887, 0.4, 0.501, 0.471, 0.511, 0.405, 0.51],
    [0.997, 0.983, 0.914, 0.993, 0.995, 0.923, 0.908, 0.922, 0.905, 0.942, 0.942, 0.908, 0.941, 0.919, 0.416, 0.523, 0.484, 0.548, 0.432, 0.552],
    [0.932, 0.822, 0.799, 0.757, 0.88, 0.804, 0.506, 0.872, 0.7, 0.781, 0.829, 0.893, 0.93, 0.933, 0.351, 0.469, 0.455, 0.48, 0.364, 0.604],
    [0.955, 0.917, 0.905, 0.903, 0.967, 0.884, 0.654, 0.837, 0.829, 0.852, 0.877, 0.943, 0.961, 0.958, 0.461, 0.531, 0.514, 0.558, 0.547, 0.689],
    [0.977, 0.936, 0.945, 0.858, 0.98, 0.912, 0.823, 0.827, 0.808, 0.848, 0.863, 0.933, 0.962, 0.957, 0.469, 0.566, 0.518, 0.589, 0.522, 0.722],
    [0.607, 0.548, 0.413, 0.285, 0.55, 0.473, 0.517, 0.582, 0.742, 0.786, 0.811, 0.881, 0.926, 0.946, 0.387, 0.502, 0.477, 0.525, 0.412, 0.601],
    [0.966, 0.928, 0.931, 0.932, 0.998, 0.885, 0.796, 0.87, 0.826, 0.828, 0.827, 0.94, 0.956, 0.963, 0.527, 0.586, 0.529, 0.595, 0.607, 0.764],
    [0.754, 0.662, 0.358, 0.282, 0.497, 0.558, 0.566, 0.766, 0.361, 0.397, 0.429, 0.811, 0.807, 0.832, 0.458, 0.554, 0.538, 0.561, 0.451, 0.532],
    [0.661, 0.596, 0.597, 0.465, 0.64, 0.659, 0.384, 0.728, 0.764, 0.831, 0.836, 0.864, 0.854, 0.874, 0.488, 0.518, 0.504, 0.513, 0.495, 0.57],
    [0.718, 0.624, 0.648, 0.508, 0.523, 0.672, 0.5, 0.622, 0.919, 0.925, 0.929, 0.831, 0.846, 0.845, 0.414, 0.48, 0.511, 0.492, 0.442, 0.585],
    [0.85, 0.656, 0.563, 0.659, 0.669, 0.757, 0.525, 0.777, 0.91, 0.934, 0.946, 0.896, 0.904, 0.869, 0.503, 0.568, 0.552, 0.558, 0.478, 0.561],
    [0.824, 0.743, 0.655, 0.647, 0.647, 0.72, 0.691, 0.687, 0.921, 0.955, 0.949, 0.929, 0.902, 0.931, 0.482, 0.527, 0.495, 0.501, 0.495, 0.505],
]

df = DataFrame(scores, index=Index, columns=Cols)

def exact_binomial_ci(successes, trials=1000, confidence_level=0.95):
    result = stats.binomtest(successes, trials)
    return result.proportion_ci(confidence_level=confidence_level, method='exact')

if __name__ == "__main__":
    # average accuracies by model
    print(df.transpose().mean())
    print(df[Cols[:8]].transpose().mean())
    print(df[Cols[8:14]].transpose().mean())
    print(df[Cols[14:]].transpose().mean())

    # average accuracies by language
    dft = df.transpose()
    dft = dft.drop('human', axis=1)
    dft['language'] = ['basque' for _ in range(8)] + ['hindi' for _ in range(6)] + ['swahili' for _ in range(6)]
    print(dft.groupby('language').mean().transpose().mean())

    cis = [[0 for _ in row] for row in scores]
    for i, row in enumerate(scores):
        for j, value in enumerate(row):
            ci = exact_binomial_ci(int(value * 1000))
            cis[i][j] = f"({ci.low:.3f}-\n{ci.high:.3f})"

    plt.figure(figsize=(20, 20))

    ax1 = sns.heatmap(df, cmap='RdYlGn', annot=True, annot_kws={'va': 'bottom', 'weight': 'bold'}, fmt='.3f', square=True, cbar=False)
    ax2 = sns.heatmap(df, cmap='RdYlGn', annot_kws={'va': 'top'}, annot=cis, fmt="", cbar=False)

    ax1.hlines(list(range(len(Index) + 1)), *ax1.get_xlim(), color="black", lw=0.5)
    ax1.hlines([1, 3, 9, 14, 15], *ax1.get_xlim(), color="black", lw=4)
    ax1.vlines(list(range(len(Cols) + 1)), *ax1.get_ylim(), color="black", lw=0.5)
    ax1.vlines([8, 14], *ax1.get_ylim(), color="black", lw=4)

    plt.xticks(rotation=70)
    plt.yticks(rotation=0)

    plt.title('Accuracy', fontsize=14)
    plt.xlabel('Test suites', fontsize=12)
    plt.ylabel('Models', fontsize=12)

    plt.savefig('results.pdf')