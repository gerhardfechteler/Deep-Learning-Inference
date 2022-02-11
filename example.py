import numpy as np
from MLP_module import MLP
import matplotlib.pyplot as plt
from keras.backend import clear_session


###############################################################################
# Generate data

# number of observations
n_obs = 1000

# regressor
X = np.random.uniform(-3,5,(n_obs,1))

# dependent variable
Y = X**3 - X**2 + np.random.normal(0,10,X.shape)


###############################################################################
# Obtain true function and true marginal effects

# evaluation points for prediction and marginal effects
X0 = np.linspace(np.min(X), np.max(X), 100).reshape((-1,1))

# true function at the evaluation points
Y_true = X0**3 - X0**2

# true marginal effects at the evaluation points
ME_true = 3*X0**2 - 2*X0


###############################################################################
# Estimate the function via the MLP_module

# create an instance of the MLP class
mymod = MLP()

# estimate the marginal effects and confidence intervals. We consider a MLP
# with 1 hidden layer and 10 neurons in the hidden layer.
ME, ME_CI_low, ME_CI_upp = mymod.compute_ME_CI_boot(X, Y, X0, w=[10], R=10)

# Obtain predictions for the function at the evaluation points
Y_pred = mymod.predict(X0)

# clear the tensorflow session.
clear_session()


###############################################################################
# Plot the results

# Enable LaTeX notation in the labels of the plot
plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

plt.figure(figsize=(10,5))

# plot the estimated function, true function and data points
plt.subplot(121)
plt.scatter(X, Y, c='k', alpha=0.15, 
            label='observed data, $n={}$'.format(n_obs))
plt.plot(X0, Y_true, '-k', label='true function')
plt.plot(X0, Y_pred, '-b', label='estimated function, $w_2=10$')
plt.ylabel('y')
plt.xlabel('x')
plt.title('Function')
plt.legend()

# Plot the estimated marginal effects, true marginal effects and confidence 
# intervals for the estimated marginal effects
plt.subplot(122)
plt.plot(X0, ME_true, '-k', label='true marginal effects')
plt.plot(X0.flatten(), ME.flatten(), '-b', label='estimated marginal effects')
plt.plot(X0.flatten(), ME_CI_low.flatten(), '--r', 
         label='estimated 95\% confidence intervals')
plt.plot(X0.flatten(), ME_CI_upp.flatten(), '--r')
plt.ylabel('Derivative / marginal effect M(x)')
plt.xlabel('x')
plt.title('Derivative / Marginal Effects')
plt.legend()

plt.tight_layout()



