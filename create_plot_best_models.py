import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt

fnames = [
    "sudden_drift_5",
    "sudden_drift_10",
    "sudden_drift_20",
    "sudden_drift_30",
    "gradual_drift_5",
    "gradual_drift_10",
    "gradual_drift_20",
    "gradual_drift_30",
]


for ind, fname in enumerate(fnames):
    results = np.load("wyniki_dobre_douczanie/Stream_%s_imbalance.npy" % fname)
    results_mse = np.load("wyniki_mse_2/Stream_%s_imbalance.npy" % fname)
    results_wo_samp = np.load("wyniki_same_wagi/Stream_%s_imbalance.npy" % fname)
    new_results = np.concatenate((results, results_mse, results_wo_samp), axis=1)

    new_results = np.mean(new_results, axis=0)
    kernel = 5
    metrics = ['f-score', 'gmean', 'bac']

    colors = [
            [
                "dodgerblue",
                "cyan",
                "darkviolet",
                "mediumslateblue",
                "green",
                "blue",
                "red"
            ],
            [
                "dodgerblue",
                "cyan",
                "darkviolet",
                "mediumslateblue",
                "green",
                "blue",
                "red"
            ],
            [
                "dodgerblue",
                "cyan",
                "darkviolet",
                "mediumslateblue",
                "green",
                "blue",
                "red"
            ],
        ]

    labels = ['u-AWE-G', 'u-AWE-B', 'u-AWE-F', 'o-AWE-G', 'o-AWE-B', 'o-AWE-F',
              'u-AUE-G', 'u-AUE-B', 'u-AUE-F', 'o-AUE-G', 'o-AUE-B', 'o-AUE-F',
              'WAE', 'OOB', 'UOB', 'AWE', 'u-AWE', 'o-AWE', 'AUE', 'u-AUE', 'o-AUE', 'AWE-G', 'AWE-B', 'AWE-F',
              'AUE-G', 'AUE-B', 'AUE-F']

    usages = [
        [9, 10, 11, 26, 12, 13, 14],
        [9, 10, 11, 26, 12, 13, 14],
        [9, 10, 11, 26, 12, 13, 14],
    ]

    lw = [
            [2, 2, 2, 2, 1, 1, 1],
            [2, 2, 2, 2, 1, 1, 1],
            [2, 2, 2, 2, 1, 1, 1],
            ]
    locs = [1, 3, 3]

    print(new_results.shape)

    fig, ax = plt.subplots(3, 1, figsize=(8, 11))
    for i in range(3):
        for j, row in enumerate(new_results[usages[i], :, i]):
            res = medfilt(row, kernel)
            ax[i].plot(res, c=colors[i][j], ls='-', label=labels[usages[i][j]], lw=lw[i][j])
        ax[i].set_title(metrics[i])
        ax[i].set_ylim(0, 1)
        ax[i].legend(ncol=3, frameon=False)

    fig.suptitle(fname)
    # plt.tight_layout()
    plt.savefig("foo.png")
    plt.savefig("figures_best_models/%s.png" % fname)


