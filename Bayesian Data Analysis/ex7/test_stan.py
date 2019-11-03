import pystan
import matplotlib.pyplot as plt

fit = pystan.stan(model_code="parameters {real theta;} model {theta ~ normal(0,1);}")

samples = fit.extract(permuted=True)
plt.hist(samples['theta'], 50)
plt.show()