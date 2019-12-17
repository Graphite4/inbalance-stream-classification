from AWE import AWE
from sklearn.naive_bayes import GaussianNB
from strlearn.streams import StreamGenerator
from sklearn.metrics import accuracy_score
from strlearn.utils.metrics import bac

from strlearn.evaluators import TestThenTrain

clf = GaussianNB()

ensemble = AWE(clf)
stream = StreamGenerator(n_chunks=30, n_drifts=1)
metrics = [accuracy_score, bac]
evaluator = TestThenTrain(metrics)
evaluator.process(stream, ensemble)
print(evaluator.scores)