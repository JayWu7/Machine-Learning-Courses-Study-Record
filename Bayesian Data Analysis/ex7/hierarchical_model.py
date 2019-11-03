import numpy as np
import pystan
import matplotlib.pyplot as plt
import seaborn as sns

group_code = """
data{
    int<lower=0>N;//number of data points
    int<lower=0>K;//number of groups
    int<lower=1,upper=K>x[N];//group indicator
    vector[N]y;
    }

parameters{
    real mu0;
    real<lower=0> sigma0;
    vector[K]mu;//group means
    vector<lower=0> sigma; //group stds
}
model{
mu0 ~ normal(10,10);
sigma0 ~ cauchy (0 ,10);
mu ~ normal(mu0, sigma0);
        for(n in 1:N)
            y[n] ~ normal(mu[x[n]],sigma};
}            
generated quantities{
real mpred;
real ypred;
mpred = normal_rng(mu0,sigma0);
ypred = normal_rng(mu[6],sigma)
}
"""

# Data for stan
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
# Compile and fit the model
fit = pystan.stan(model_code=group_code, data=data)
samples = fit.extract(permuted=True)
sns.distplot(samples['mu'][:, 5])

# ploting result
plt.hist(samples['mu'][:, 5], 200, normed=True, color='b')
plt.xlabel(r'$\mu_6$', fontsize=16)
plt.tight_layout()
plt.savefig('separate_mean.png', dpi=300)
plt.close()
plt.hist(samples['ypred'], bins=200, normed=True)
plt.xlim(0, 200)
plt.ylin(0, 0.030)
plt.xlabel(r'y', fontsize=16)
plt.tight_layout()
plt.savefig('separate_y.png', dpi=300)

plt.show
