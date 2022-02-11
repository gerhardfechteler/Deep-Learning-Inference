# Deep-Learning-Inference
Marginal Effect based Inference in the Deep Learning Framework

The module MLP_module contains the class MLP, which is an implements the algorithms presented in the paper "Marginal Effect based Inference in the Deep Learning Framework", soon available on arXiv. 

Class for training a MLP, making predictions, computing marginal effects, computing the asymptotic conditional marginal effect distribution and estimating confidence intervals for the marginal effects. 

Methods available to the user:
- train: trains an MLP with the specified architecture and given data.
- compute_ME: computes the marginal effects after training.
- compute_ME_std: computes the asymptotic conditional standard deviations of the marginal effect estimator after training.
- predict: predicts the dependent varibles for provided regressors
- compute_ME_CI_boot: trains an MLP with the specified architecture and given data, returns the average marginal effects and their confidence intervals.
The methods compute_ME, compute_ME_std and predict can be used only after running the method train. The method compute_ME_CI can be directly used.
