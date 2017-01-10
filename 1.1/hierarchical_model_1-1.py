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
  N = 15
  D = 30
  exp_scale = 10
  sigma = 1
  observations = np.zeros((N, D))
  mus = np.random.exponential(scale=10, size=N)
  for i, mu in enumerate(mus):
    observations[i] = np.random.normal(loc=mu, scale=sigma, size=D)
  print "Actual exp_scale: %s"%exp_scale
  print "Actual mus: %s"%mus
  print "Actual sigma: %s"%sigma


  ## Your code here
  ###################################################

  # Construct dictionary to pass to Stan
  dataset = {
    'N': N,
    'D': D,
    'observations': observations,
  }
  return dataset


def inference(dataset):
  print '\tPerforming inference...'
  NUM_ITER = 1000
  WARMUP = 500
  NUM_CHAINS = 4
  NUM_CORES = 4
  STAN_FN = 'hierarchical_model_1-1.stan'

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
