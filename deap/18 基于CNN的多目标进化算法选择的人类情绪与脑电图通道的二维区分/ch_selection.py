from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_reference_directions
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.factory import get_termination
import warnings
from ch_problem import ChSelectionDef
from utils import get_dataset, check_folders
from multiprocessing.pool import ThreadPool
import pickle
import string
import random
warnings.filterwarnings("ignore")
from pathlib import Path
import os

def get_algorithm(_method='NSGA2', no_chromo=20):
  if _method=='NSGA2':
    algorithm = NSGA2(pop_size=no_chromo,sampling=get_sampling("bin_random"),crossover=get_crossover("bin_two_point"),mutation=get_mutation("bin_bitflip"),elimate_duplicates=True)
  elif _method=='NSGA3':
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=no_chromo)# , n_obj,
    algorithm = NSGA3(pop_size=no_chromo,
      ref_dirs=ref_dirs,
      sampling=get_sampling("bin_random"),
      crossover=get_crossover("bin_two_point"),
      mutation=get_mutation("bin_bitflip"),
      elimate_duplicates=True)
  return algorithm

def get_termination_f(_t="n_gen", no_generations=200):
  if _t=="n_gen":
    termination = get_termination("n_gen", no_generations)
  elif _t=="f_tol":
    termination = get_termination("f_tol",
      tol=0.001,
      n_last=10,
      n_max_gen=no_generations,
      nth_gen=5)
  return termination

def start_optimization(_step,_subject,type_class,epochs_fit,n_items,no_chromo, no_generations):
  print(_subject,type_class, "+*"*50)
  str_rand = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(11))
  _result_opt = {}
  _method_txt = 'NSGA2'
  dataset = get_dataset(_step, _subject, type_class)
  problem = ChSelectionDef(_subject,dataset,epochs_fit,n_items,no_chromo)
  algorithm = get_algorithm(_method_txt, no_chromo)
  termination = get_termination_f("f_tol", no_generations)
  _result_opt[str_rand] = minimize(problem,
                 algorithm,
                 termination,
                 seed=1,
                 verbose=False,
                 save_history=True)
  print('------------------------***********************'*2)
  print(_subject, type_class, "++"*50)
  print("Saving file","+*"*100)
  f_name = 'results_opt/ch_results_{0}_{1}_{2}_{3}_{4}.pkl'.format(_subject,_step,type_class,no_chromo,no_generations)
  try:
    if Path(f_name).is_file():
      print("IT EXIST", "Removing...", "++"*50)
      os.remove(f_name)
    else:
      pickle.dump(_result_opt[str_rand], open(f_name, 'wb'), protocol=2)
      print("Saved file","+*"*100)
  except Exception as e:
    if Path(f_name).is_file():
      print("IT EXIST", "Removing...", "++"*50)
      os.remove(f_name)
    else:
      pickle.dump(_result_opt[str_rand], open("results_e/"+f_name, 'wb'), protocol=2)
      print("Saved file","+*"*100)

check_folders()
n_items = 32
no_chromo = 10
no_generations = 100 #200
epochs_fit = 3#200
_step = 2 # [10,5,2] # 3, 60
type_classes = ["Arousal", "Valence"]
params_1 = [[_step,_subject,
            type_class, epochs_fit,
            n_items, no_chromo, 
            no_generations]
            for type_class in type_classes
            for _subject in range(1,3) # 33
            ]
pool_1 = ThreadPool(len(params_1))
pool_1.starmap(start_optimization, params_1)
