import numpy as np
import autograd.numpy as anp
from pymoo.core.problem import Problem
from multiprocessing.pool import ThreadPool
from utils import train_model, test_training
import string
import random

def get_accuracy(subject, dataset, chromo_chs, epochs_fit):
  """
  Get the accuracy for each chromosome
  """
  matFeats = []
  for trial in dataset["data"]:
      _segment = np.array([trial[k] for k,ch in enumerate(chromo_chs) if ch==1])
      matFeats.append(_segment)
  matFeats = np.array(matFeats)
  X, X_validate, X_test, y, y_validate, y_test = test_training(matFeats, dataset["tags"])
  s_ran = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
  _acc = train_model(subject, s_ran, X, X_validate, X_test, y, y_validate, y_test, epochs_fit)
  return _acc

def func_worker(chromosome, subject, dataset, epochs_fit):
  no_chs = sum(chromosome)
  if no_chs>=1:
    _acc = get_accuracy(subject, dataset, chromosome, epochs_fit)
  else:
    _acc = 0
  print("no_chs", no_chs, "_acc", _acc)
  return no_chs, _acc

class ChSelectionDef(Problem):
    def __init__(self, subject, dataset, epochs_fit, n_items, no_chromo):
        self.subject = subject
        self.dataset = dataset
        self.epochs_fit = epochs_fit
        self.no_chromo = no_chromo
        n_constr = 0
        n_obj = 2
        super().__init__(n_var=n_items, n_obj=n_obj, n_constr=n_constr, xl=0, xu=1, type_var=bool)
        self.n_var = n_items
        self.n_constr = n_constr
        self.n_obj = n_obj
        self.func = self._evaluate

    def _evaluate(self, x, out, *args, **kwargs):
      print('Population, chromosomes group---------------------------------------------------')
      f1 = np.array([])
      f2 = np.array([])
      params = [[x[k], self.subject, self.dataset, self.epochs_fit] for k in range(len(x))]
      pool = ThreadPool(self.no_chromo)
      jobs = pool.starmap(func_worker, params)
      for job in jobs:
        no_chs, _acc = job
        f1 = np.concatenate((f1, [no_chs]))
        f2 = np.concatenate((f2, [_acc]))
      out["F"] = anp.column_stack([f1, -f2])
