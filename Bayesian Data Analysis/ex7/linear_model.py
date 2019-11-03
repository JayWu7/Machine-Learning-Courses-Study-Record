import numpy as np
import matplotlib.pyplot as plt
import pystan
import seaborn as sns


def model():
    linear_code = """ 
data{
    int<lower = 0>N;//number of data points
    vector[N]x;//
    vector[N]y;//
    real xpred;//input location for prediction
}

parameters{
    real alpha;
    real beta;
    real<lower=0>sigma;
}
transformed parameters{
    vector[N]mu;
    mu<-alpha+beta*x;
}
model{
    y~normal(mu,sigma);
}
generated quantities{
    real ypred;
    vector[N]log_lik;
    ypred<-normal_rng(alpha+beta*xpred,sigma);
    for(n in 1:N)
        log_lik[n]<-normal_log(y[n],alpha+beta*x[n],sigma);}
    """
    data_path = 'drowning.txt'
    d = np.loadtxt(data_path, dtype=np.double, skiprows=0)
    x = d[:, 0]
    y = d[:, 1]
    N = len(x)
    xpred = 2019
    data = dict(N=N, x=x, y=y, xpred=xpred)

    fit = pystan.stan(model_code=linear_code, data=data)
    samples = fit.extract(permuted=True)
    plt.figure(figsize=(8, 10))
    plt.subplot(3, 1, 1)
    plt.plot(x, np.percentile(samples['mu'], 50, axis=0), color='red')
    plt.plot(x, np.asarray(np.percentile(samples['mu'], [5, 95], axis=0)).T, linestyle='--', color='red')
    plt.scatter(x, y, 5, color='black')
    plt.xlabel('Year')
    plt.ylabel('Number of drownings')
    plt.xlim((1975, 2020))
    plt.subplot(3, 1, 2)
    plt.hist(samples['beta'], bins=50, alpha=0.8, rwidth=0.95)
    plt.xlabel('beta')
    plt.subplot(3, 1, 3)
    plt.hist(samples['ypred'], 50, alpha=0.8, rwidth=0.95)
    plt.xlabel('posterior predictive histogram for year {}'.format(xpred))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    model()
