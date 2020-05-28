from AWE import AWE
from sklearn.naive_bayes import GaussianNB
from Classifier_wrapper import HoeffdingTreeWrapper
from strlearn.streams import StreamGenerator
from strlearn.ensembles import UOB, OOB, WAE
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,  balanced_accuracy_score as bac
from imblearn.metrics import geometric_mean_score as g_mean

from strlearn.evaluators import TestThenTrain

import matplotlib.pyplot as plt
import numpy as np


class Experiment:

    def __init__(self, streams_random_seeds=None, ensembles=(), ensembles_labels=(), metrics=None,
                 imbalance=None, gradual=True, n_chunks=100):
        if metrics is None:
            metrics = [accuracy_score]
        if imbalance is None:
            imbalance = [0.50, 0.50]
        self._streams_random_seeds = streams_random_seeds
        self._ensembles = ensembles
        self._metrics = metrics
        self._ensembles_labels = ensembles_labels
        self._proportions = imbalance
        self._gradual_drift = gradual
        self.n_chunks = n_chunks
        self._evaluator = TestThenTrain(self._metrics)
        self._scores = np.empty((len(self._streams_random_seeds), len(self._ensembles), self.n_chunks-1, len(self._metrics)))

    def conduct(self, file=None):
        for r_i, r in enumerate(self._streams_random_seeds):

            if self._gradual_drift:
                stream = StreamGenerator(n_chunks=self.n_chunks, chunk_size=500, n_drifts=5, weights=self._proportions,
                                         n_features=2,
                                         n_informative=2, n_redundant=0, n_repeated=0, random_state=r, concept_sigmoid_spacing=5)
            else:
                stream = StreamGenerator(n_chunks=self.n_chunks, chunk_size=500, n_drifts=5, weights=self._proportions, n_features=2,
                                     n_informative=2, n_redundant=0, n_repeated=0, random_state=r)

            self._evaluator.process(stream, self._ensembles)

            self._scores[r_i, :, :, :] = self._evaluator.scores[:, :, :]

        if file is not None:
            np.save(file, self._scores)


    def make_plot(self, title):
        results = np.mean(self._scores, axis=0)
        metrics_labels = [metric.__name__ for metric in self._metrics]
        for i in range(len(self._metrics)):
            for j in range(len(self._ensembles)):
                plt.plot(results[j, :, i], label=self._ensembles_labels[j])

            plt.title(metrics_labels[i])
            plt.ylim(0, 1)
            plt.ylabel('Jakość')
            plt.xlabel('Chunk')

            plt.legend()
            plt.savefig(title + '_' + metrics_labels[i] + ".png")
            plt.clf()



if __name__ == '__main__':
    clf = HoeffdingTreeWrapper()

    random_seeds = [4, 13, 42, 44, 666]

    imbalance = [0.02]

    ensemble1 = AWE(clf, 10, "proportional_to_mse")
    ensemble2 = AWE(clf, 10, "proportional_to_mse", sampling="under")
    ensemble3 = AWE(clf, 10, "proportional_to_mse", sampling="over")
    # ensemble4 = AWE(clf, 10, "proportional_to_g-mean", sampling="over")
    # ensemble5 = AWE(clf, 10, "proportional_to_bac", sampling="over")
    # ensemble6 = AWE(clf, 10, "proportional_to_f1", sampling="over")
    ensemble7 = AWE(clf, 10, "proportional_to_mse", update=True)
    ensemble8 = AWE(clf, 10, "proportional_to_mse", sampling="under", update=True)
    ensemble9 = AWE(clf, 10, "proportional_to_mse", sampling="over", update=True)
    # ensemble10 = AWE(clf, 10, "proportional_to_g-mean", sampling="over", update=True)
    # ensemble11 = AWE(clf, 10, "proportional_to_bac", sampling="over", update=True)
    # ensemble12 = AWE(clf, 10, "proportional_to_f1", sampling="over", update=True)
    # ensemble13 = WAE(clf, 10)
    # ensemble14 = OOB(clf, 10)
    # ensemble15 = UOB(clf, 10)
    # ensembles = (ensemble1, ensemble2, ensemble3, ensemble4, ensemble5, ensemble6, ensemble7, ensemble8, ensemble9,
    #              ensemble10, ensemble11, ensemble12, ensemble13, ensemble14, ensemble15)
    # ensemble_labels = [ensembles[i].get_name() for i in range(12)] + ['WAE', 'OOB', 'UOB']
    ensembles = (ensemble1, ensemble2, ensemble3, ensemble7, ensemble8, ensemble9)
    ensemble_labels = [e.get_name() for e in ensembles]

    path = 'wyniki_mse_2/'
    for i in imbalance:
        e = Experiment(streams_random_seeds=random_seeds, ensembles=ensembles, ensembles_labels=ensemble_labels,
                       metrics=[f1_score, g_mean, bac], imbalance=[1 - i, i], gradual=False)
        e.conduct(file=path+'Stream_sudden_drift_'+str(int(100*i))+'_imbalance.npy')

        e = Experiment(streams_random_seeds=random_seeds, ensembles=ensembles, ensembles_labels=ensemble_labels,
                       metrics=[f1_score, g_mean, bac], imbalance=[1 - i, i], gradual=True)
        e.conduct(file=path+'Stream_gradual_drift_'+str(int(100*i))+'_imbalance.npy')

    ensemble1 = AWE(clf, 10, "proportional_to_g-mean", sampling="under")
    ensemble2 = AWE(clf, 10, "proportional_to_bac", sampling="under")
    ensemble3 = AWE(clf, 10, "proportional_to_f1", sampling="under")
    ensemble4 = AWE(clf, 10, "proportional_to_g-mean", sampling="over")
    ensemble5 = AWE(clf, 10, "proportional_to_bac", sampling="over")
    ensemble6 = AWE(clf, 10, "proportional_to_f1", sampling="over")
    ensemble7 = AWE(clf, 10, "proportional_to_g-mean", sampling="under", update=True)
    ensemble8 = AWE(clf, 10, "proportional_to_bac", sampling="under", update=True)
    ensemble9 = AWE(clf, 10, "proportional_to_f1", sampling="under", update=True)
    ensemble10 = AWE(clf, 10, "proportional_to_g-mean", sampling="over", update=True)
    ensemble11 = AWE(clf, 10, "proportional_to_bac", sampling="over", update=True)
    ensemble12 = AWE(clf, 10, "proportional_to_f1", sampling="over", update=True)
    ensemble13 = WAE(clf, 10)
    ensemble14 = OOB(clf, 10)
    ensemble15 = UOB(clf, 10)
    ensembles = (ensemble1, ensemble2, ensemble3, ensemble4, ensemble5, ensemble6, ensemble7, ensemble8, ensemble9,
                 ensemble10, ensemble11, ensemble12, ensemble13, ensemble14, ensemble15)
    ensemble_labels = [ensembles[i].get_name() for i in range(12)] + ['WAE', 'OOB', 'UOB']

    path = 'wyniki_dobre_douczanie/'
    for i in imbalance:
        e = Experiment(streams_random_seeds=random_seeds, ensembles=ensembles, ensembles_labels=ensemble_labels,
                       metrics=[f1_score, g_mean, bac], imbalance=[1 - i, i], gradual=False)
        e.conduct(file=path + 'Stream_sudden_drift_' + str(int(100 * i)) + '_imbalance.npy')

        e = Experiment(streams_random_seeds=random_seeds, ensembles=ensembles, ensembles_labels=ensemble_labels,
                       metrics=[f1_score, g_mean, bac], imbalance=[1 - i, i], gradual=True)
        e.conduct(file=path + 'Stream_gradual_drift_' + str(int(100 * i)) + '_imbalance.npy')

