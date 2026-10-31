from copy import deepcopy
from itertools import combinations, permutations
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import MultivariateNormal, kl_divergence
from torch.nn import functional as F

from pl_staging.model import (ProbabilisticAE, ascending_pl_nll,
                              joint_gaussian_kl, loss_terms, score_model)


torch.set_num_threads(1)


def make_model(**overrides):
    config = dict(hidden_dims=[8, 4], activation='gelu', dropout=0., seed=42,
                  beta=.7, tau=.3, rho=.4, rank_samples=16)
    config.update(overrides)
    return ProbabilisticAE(3, config)


@pytest.mark.parametrize('rho', [0., .4, .95, .9999])
def test_joint_kl_values_and_gradients_match_independent_torch_formula(rho):
    mu = torch.tensor([.8, -.2, .3, 1.1, -.6, .4], dtype=torch.float64, requires_grad=True)
    logvar = torch.tensor([-.7, .3, -.1, .8, -.4, .2], dtype=torch.float64, requires_grad=True)
    indices = torch.tensor([[0, -1, -1], [1, 2, -1], [3, 4, 5]])
    mask = indices >= 0
    actual = joint_gaussian_kl(mu, logvar, indices, mask, rho)
    expected = []
    for row in indices:
        ix = row[row >= 0]
        size = len(ix)
        covariance = (1-rho) * torch.eye(size, dtype=mu.dtype) + rho * torch.ones((size, size), dtype=mu.dtype)
        q = MultivariateNormal(mu[ix], covariance_matrix=torch.diag(logvar[ix].exp()))
        prior = MultivariateNormal(torch.zeros(size, dtype=mu.dtype), covariance_matrix=covariance)
        expected.append(kl_divergence(q, prior))
    expected = torch.stack(expected)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)
    ag = torch.autograd.grad(actual.sum(), (mu, logvar), retain_graph=True)
    eg = torch.autograd.grad(expected.sum(), (mu, logvar))
    for left, right in zip(ag, eg):
        torch.testing.assert_close(left, right, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize('size', [1, 2, 3, 4])
def test_complete_permutation_probabilities_sum_to_one(size):
    z = torch.tensor([-.8, .4, 1.2, -.1], dtype=torch.float64)[:size]
    ordered = torch.stack([z[list(order)] for order in permutations(range(size))])
    mask = torch.ones_like(ordered, dtype=torch.bool)
    nll = ascending_pl_nll(ordered, mask, .7)
    torch.testing.assert_close((-nll).exp().sum(), torch.tensor(1., dtype=z.dtype), rtol=1e-13, atol=1e-13)


def test_two_visits_equal_softplus_and_padding_has_zero_gradient():
    z = torch.tensor([[.7, -.4, 1.e10], [.2, -1.e10, 1.e10]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    actual = ascending_pl_nll(z, mask, .2)
    expected = torch.stack((F.softplus((z[0, 0]-z[0, 1])/.2), z[1, 0]*0.))
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual.sum(), z)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[0, 0] > 0 and gradient[0, 1] < 0
    assert torch.equal(gradient[~mask], torch.zeros_like(gradient[~mask]))
    assert gradient[1, 0] == 0


def test_pl_shift_invariance_gradient_and_extreme_temperature():
    z = torch.tensor([[2., -.2, .1], [-3., 1., 1.2]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones_like(z, dtype=torch.bool)
    first = ascending_pl_nll(z, mask, .03)
    shifted = ascending_pl_nll(z + torch.tensor([[1000.], [-500.]], dtype=z.dtype), mask, .03)
    torch.testing.assert_close(first, shifted, atol=2e-11, rtol=2e-12)
    gradient = torch.autograd.grad(first.sum(), z, retain_graph=True)[0]
    torch.testing.assert_close(gradient.sum(dim=-1), torch.zeros(2, dtype=z.dtype), atol=1e-12, rtol=0)
    assert torch.autograd.gradcheck(lambda value: ascending_pl_nll(value, mask, .03), (z,), eps=1e-6)
    extreme = torch.tensor([[-1.e5, 0., 1.e5], [1.e5, 0., -1.e5]], dtype=torch.float64, requires_grad=True)
    result = ascending_pl_nll(extreme, mask, 1.e-5)
    assert torch.isfinite(result).all() and (result >= 0).all()
    assert torch.isfinite(torch.autograd.grad(result.sum(), extreme)[0]).all()


class FixedPosterior(nn.Module):
    def __init__(self, means, logvars, tau=.4, rho=.3, rank_samples=16):
        super().__init__()
        self.mu = nn.Parameter(torch.as_tensor(means, dtype=torch.float64))
        self.logvar = nn.Parameter(torch.as_tensor(logvars, dtype=torch.float64))
        self.input_dim = 1
        self.tau = tau
        self.rho = rho
        self.beta = .7
        self.rank_samples = rank_samples

    def posterior(self, x):
        return self.mu, self.logvar

    def decode(self, z):
        return z[:, None]


def test_antithetic_estimator_and_normalization_reproduce_explicit_draws():
    model = FixedPosterior([-.3, .8, .2], [-.2, .3, -.6])
    x = torch.tensor([[.4], [-.2], [.7]], dtype=torch.float64)
    indices = torch.tensor([[0, 1], [2, -1]])
    mask = indices >= 0
    normalizer = 2 * 7 / 3
    torch.manual_seed(91)
    terms = loss_terms(model, x, indices, mask, normalizer)
    torch.manual_seed(91)
    std = model.logvar.mul(.5).exp()
    rec_z = model.mu + std * torch.randn_like(model.mu)
    eps = torch.randn((8, 3), dtype=x.dtype)
    z = model.mu + std * torch.cat((eps, -eps))
    expected_rank = F.softplus((z[:, 0]-z[:, 1])/model.tau).mean()/normalizer
    torch.testing.assert_close(terms['rank'], expected_rank)
    torch.testing.assert_close(terms['rec'], (rec_z[:, None]-x).square().sum()/normalizer)
    expected_kl = joint_gaussian_kl(model.mu, model.logvar, indices, mask, model.rho).sum()/normalizer
    torch.testing.assert_close(terms['kl'], expected_kl)
    total = terms['rec'] + model.beta*(terms['rank']+terms['kl'])
    assert torch.isfinite(torch.autograd.grad(total, (model.mu, model.logvar))[0]).all()


def test_uniform_subject_normalizer_has_exact_full_objective_expectation():
    # Exact enumeration, rather than a noisy simulation: visit counts differ.
    losses = torch.tensor([.7, 2.1, 8.4, 1.2], dtype=torch.float64)
    counts = torch.tensor([1, 2, 5, 1])
    n, total_scans, batch = len(counts), counts.sum().item(), 2
    estimates = [losses[list(ix)].sum()/(batch*total_scans/n) for ix in combinations(range(n), batch)]
    torch.testing.assert_close(torch.stack(estimates).mean(), losses.sum()/total_scans)
    ratio_estimates = [losses[list(ix)].sum()/counts[list(ix)].sum() for ix in combinations(range(n), batch)]
    assert not torch.isclose(torch.stack(ratio_estimates).mean(), losses.sum()/total_scans)


def test_expected_two_visit_rank_matches_independent_gaussian_quadrature():
    # Quadrature is a test reference only; training uses antithetic Monte Carlo.
    model = FixedPosterior([-.2, .6], [-1., -.3], rank_samples=65536)
    torch.manual_seed(42)
    terms = loss_terms(model, torch.zeros((2, 1), dtype=torch.float64), [[0, 1]], [[True, True]], 1.)
    nodes, weights = np.polynomial.hermite.hermgauss(96)
    difference = -.2-.6
    variance = math.exp(-1.)+math.exp(-.3)
    expected = np.dot(weights, np.logaddexp(0., (difference+np.sqrt(2*variance)*nodes)/model.tau))/np.sqrt(np.pi)
    assert abs(terms['rank'].item()-expected) < .015


def test_singleton_standard_normal_and_invalid_partition():
    model = FixedPosterior([0., 0.], [0., 0.], rho=.99)
    terms = loss_terms(model, torch.zeros((2, 1), dtype=torch.float64), [[0], [1]], [[True], [True]], 2.)
    assert terms['rank'].item() == 0. and terms['kl'].item() == 0.
    assert torch.isfinite(torch.autograd.grad(terms['rank'], model.mu)[0]).all()
    with pytest.raises(ValueError, match='partition'):
        loss_terms(model, torch.zeros((2, 1), dtype=torch.float64), [[0], [0]], [[True], [True]], 2.)


def test_score_is_encoder_expectation_and_never_uses_decoder():
    model = make_model()
    model.train()
    x = np.random.default_rng(42).normal(size=(4, 3))
    reference = deepcopy(model).double().eval()
    with torch.no_grad():
        mu, logvar = reference.posterior(torch.from_numpy(x))
    def forbidden_decode(*args, **kwargs):
        raise AssertionError('Stage scoring must not use the decoder')
    model.decode = forbidden_decode
    result = score_model(model, x)
    np.testing.assert_array_equal(result['stage_mean'], mu.numpy())
    np.testing.assert_array_equal(result['stage_logvar'], logvar.numpy())
    np.testing.assert_array_equal(result['stage_variance'], logvar.exp().numpy())
    assert result['diagnostics']['uses_decoder'] is False
    assert model.training and next(model.parameters()).dtype == torch.float32


def test_defaults_clamping_and_initialization_are_reproducible():
    first, second = make_model(), make_model()
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    with torch.no_grad():
        first.to_logvar.weight.zero_()
        first.to_logvar.bias.fill_(100.)
    assert torch.equal(first.posterior(torch.zeros((2, 3)))[1], torch.full((2,), 6.))
    for overrides in ({'tau': 0}, {'rho': 1}, {'rank_samples': 3}, {'beta': -1}, {'dropout': .1}):
        with pytest.raises(ValueError):
            make_model(**overrides)
