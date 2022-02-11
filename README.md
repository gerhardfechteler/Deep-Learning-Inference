# Deep-Learning-Inference

The module MLP_module implements the algorithms presented in the paper "Marginal Effect based Inference in the Deep Learning Framework", soon available on arXiv. The link will be added here.

MLP_module contains the class MLP for training a multilayer perceptron, making predictions, computing the derivatives of outputs with respect to inputs, usually denoted as marginal effects or sensitivities, and estimating confidence intervals for the marginal effects. 

Methods available to the user:
- train: trains an MLP with the specified architecture and given data.
- compute_ME: computes the marginal effects after training.
- compute_ME_std: computes the asymptotic conditional standard deviations of the marginal effect estimator after training.
- predict: predicts the dependent varibles for provided regressors
- compute_ME_CI_boot: trains an MLP with the specified architecture and given data, returns the average marginal effects and their confidence intervals.

The methods compute_ME, compute_ME_std and predict can be used only after running the method train. The method compute_ME_CI can be used directly.

The file example.py shows how to use the module.
