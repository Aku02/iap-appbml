import sys, math
import numpy as np
import pystan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def simulate():
  print '\tGenerating simulated data...'
  N = 20
  D = 15
  L = 10
  features = np.random.normal(loc=0, scale=1, size=(N, L))
  weights = np.arange(-L/2, L/2)
  mu = features.dot(weights)
  sigma_mu = 5
  sigma_sigma = 1
  sigma = np.abs(np.random.normal(loc=sigma_mu, scale=sigma_sigma, size=N))
  observations = np.zeros((N, D))
  for i in range(N):
    observations[i] = np.random.normal(loc=mu[i], scale=sigma[i], size=D)
  dataset = {
    'N': N,
    'D': D,
    'L': L,
    'observations': observations,
    'features': features,
  }
  print "Actual weights: %s"%weights
  print "Actual sigma_mu: %s"%sigma_mu
  print "Actual sigma_sigma: %s"%sigma_sigma
  return dataset


def inference(dataset):
  print '\tPerforming inference...'
  NUM_ITER = 2000
  WARMUP = 200
  NUM_CHAINS = 4
  NUM_CORES = 4
  STAN_FN = 'model_1-3.stan'

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
