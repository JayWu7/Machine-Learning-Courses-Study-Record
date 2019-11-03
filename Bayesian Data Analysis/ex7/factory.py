import numpy as np
import matplotlib.pyplot as plt
import pystan
import seaborn as sns


def separate_model():
    group_code = """
    data {
        int<lower=0> N; // number of data points 
        int<lower=0> K; // number of groups 
        int<lower=1,upper=K> x[N]; // group indicator 
        vector[N] y; //
    }
    parameters {
        vector [K] mu; // group means
        real<lower=0> sigma; // common stds 
    }
    model {
        for (n in 1:N)
            y[n] ~ normal(mu[x[n]] , sigma);
    }
    generated quantities {
    real ypred;
    ypred = normal_rng(mu[6] , sigma);
    }
    """

    data_path = 'factory.txt'
    d = np.loadtxt(data_path, dtype=np.double)
    x = np.tile(np.arange(1, d.shape[1] + 1), d.shape[0])
    y = d.ravel()
    N = len(x)
    data = dict(
        N=N,
        K=6,
        x=x,
        y=y
    )
    plt.figure(figsize=(10, 10))
    fit = pystan.stan(model_code=group_code, data=data)
    samples = fit.extract(permuted=True)
    plt.subplot(3, 1, 1)
    sns.distplot(samples['mu'][:, 5])
    plt.xlabel('Mean of Machine 6')
    plt.tight_layout()
    plt.subplot(3, 1, 2)
    sns.distplot(samples['ypred'])
    plt.xlabel('Y-Predictions')
    plt.tight_layout()
    plt.subplot(3, 1, 3)
    sns.distplot(samples['mu'][:, 0])
    plt.xlabel('Mean of Machine 7')
    plt.tight_layout()
    plt.show()


# separate_model()

def pooled_model():
    group_code = """
        data {
            int<lower=0> N; // number of data points 
            vector[N] y; //
        }
        parameters {
            real mu;
            real<lower=0> sigma; // common stds 
        }
        model {
            for (n in 1:N)
                y[n] ~ normal(mu, sigma);
        }
        generated quantities {
        real ypred;
        ypred = normal_rng(mu, sigma);
        }"""
    data_path = 'factory.txt'
    d = np.loadtxt(data_path, dtype=np.double)
    y = d.ravel()
    N = len(y)
    data = dict(
        N=N,
        y=y
    )
    plt.figure(figsize=(10, 10))
    fit = pystan.stan(model_code=group_code, data=data)
    samples = fit.extract(permuted=True)
    plt.subplot(3, 1, 1)
    sns.distplot(samples['mu'])
    plt.xlabel('Mean of Machine 6')
    plt.tight_layout()
    plt.subplot(3, 1, 2)
    sns.distplot(samples['ypred'])
    plt.xlabel('Y-Predictions')
    plt.tight_layout()
    plt.subplot(3, 1, 3)
    sns.distplot(samples['mu'])
    plt.xlabel('Mean of Machine 7')
    plt.tight_layout()
    plt.show()


# pooled_model()

def hierarchical_model():
    group_code = """
        data {
            int<lower=0> N; // number of data points 
            int<lower=0> K; // number of groups 
            int<lower=1,upper=K> x[N]; // group indicator 
            vector[N] y; //
        }
        parameters {
            real mu0;
            real<lower=0> sigma0;
            vector [K] mu; // group means
            real<lower=0> sigma; // common stds 
        }
        model {
            mu0 ~ normal(10,10);
            sigma0 ~ cauchy(0,10);
            mu ~ normal(mu0, sigma0);
            for (n in 1:N)
                y[n] ~ normal(mu[x[n]] , sigma);
        }
        generated quantities {
        real mpred;
        real ypred;
        mpred = normal_rng(mu0,sigma0);
        ypred = normal_rng(mu[6] , sigma);
        }"""
    data_path = 'factory.txt'
    d = np.loadtxt(data_path, dtype=np.double)
    x = np.tile(np.arange(1, d.shape[1] + 1), d.shape[0])
    y = d.ravel()
    N = len(x)
    data = dict(
        N=N,
        K=6,
        x=x,
        y=y
    )
    plt.figure(figsize=(10, 10))
    fit = pystan.stan(model_code=group_code, data=data)
    samples = fit.extract(permuted=True)
    plt.subplot(3, 1, 1)
    sns.distplot(samples['mu'][:, 5])
    plt.xlabel('Mean of Machine 6')
    plt.tight_layout()
    plt.subplot(3, 1, 2)
    sns.distplot(samples['ypred'])
    plt.xlabel('Y-Predictions')
    plt.tight_layout()
    plt.subplot(3, 1, 3)
    sns.distplot(samples['mpred'])
    plt.xlabel('Mean of Machine 7')
    plt.tight_layout()
    plt.show()


hierarchical_model()
