import numpy as np
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

for fname in fnames:
    results = np.load("wyniki_dobre_douczanie/Stream_%s_imbalance.npy" % fname)

    results = np.mean(results, axis=0)
    kernel = 5
    metrics = ["f1", "gmean", "bac"]
    colors = ["red", "red", "blue", "blue", "green", "red", "blue"]
    ls = ["-", ":", "-", ":", "-", "-", "-"]
    labels = ["AWE RUS", "AWE ROS", "AUE RUS", "AUE ROS", "WAE", "OOB", "UOB"]

    usages = [
        [0, 3, 6, 9, 12, 13, 14],
        [1, 4, 7, 10, 12, 13, 14],
        [2, 5, 8, 11, 12, 13, 14],
    ]
    lw = [2, 2, 2, 2, 0.25, 0.25, 0.25]
    locs = [1, 3, 3]

    print(results.shape)

    fig, ax = plt.subplots(3, 1, figsize=(8, 11))
    for i in range(3):
        for j, row in enumerate(results[usages[i], :, i]):
            res = medfilt(row, kernel)
            ax[i].plot(res, c=colors[j], ls=ls[j], label=labels[j], lw=lw[j])
        ax[i].set_title(metrics[i])
        ax[i].set_ylim(0, 1)
        ax[i].legend(ncol=3, frameon=False)

    fig.suptitle(fname)
    # plt.tight_layout()
    plt.savefig("foo.png")
    plt.savefig("figures/%s.png" % fname)
