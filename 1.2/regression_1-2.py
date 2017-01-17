import sys, math
import numpy as np
import pystan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def simulate():
  print '\tGenerating simulated data...'

  ###################################################
  ## Your code here
  N = 30
  D = 10
  weights = np.arange(-5, 5)
  sigma = 3
  features = np.random.normal(loc=0, scale=1, size=(N,D))
  observations = np.random.normal(loc=features.dot(weights), scale=sigma)
  print "Actual weights: %s"%weights
  print "Actual sigma: %s"%sigma
  ## Your code here
  ###################################################

  # Construct dictionary to pass to Stan
  dataset = {
    'N': N,
    'D': D,
    'observations': observations,
    'features': features
  }
  return dataset


def inference(dataset):
  print '\tPerforming inference...'
  NUM_ITER = 2000
  WARMUP = 200
  NUM_CHAINS = 4
  NUM_CORES = 4
  STAN_FN = 'regression_1-2.stan'

  # import pdb; pdb.set_trace()
  fit = pystan.stan(file = STAN_FN, 
                    data = dataset, 
                    iter = NUM_ITER, 
                    warmup = WARMUP, 
                    chains = NUM_CHAINS, 
                    n_jobs = NUM_CORES)
  print(fit)

  fit.plot()
  plt.tight_layout()
  plt.savefig('fit_pystan.png')
  return


def main():
  dataset = simulate()
  inference(dataset)
  return


if __name__ == '__main__':
  main()
