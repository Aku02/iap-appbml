data {
  int N;
  int D;
  int L;
  matrix[N, D] observations;
  matrix[N, L] features;
}

parameters {
  vector[L] weights;
  vector<lower=0>[N] sigma;
  real sigma_mu;
  real sigma_sigma;
}

transformed parameters {
  vector[N] mu = features * weights;
}

model {
  for (i in 1:N) {
    observations[i] ~ normal(mu[i], sigma[i]);
  }
  sigma ~ normal(sigma_mu, sigma_sigma);
  sigma_mu ~ normal(0, 100);
  sigma_sigma ~ normal(0, 100);
}