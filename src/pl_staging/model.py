"""Gaussian PET staging with the full ascending Plackett--Luce likelihood.

The reported stage is the expectation of the single-PET variational Gaussian.
Training uses complete subject sequences and a correlated Gaussian stage prior.
"""
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ProbabilisticAE(nn.Module):
    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        self.config = dict(config)
        self.input_dim = int(input_dim)
        dims = list(config.get('hidden_dims', [64, 32, 16]))
        if self.input_dim < 1 or not dims or any(int(d) != d or d < 1 for d in dims):
            raise ValueError('Input dimension and hidden widths must be positive integers')
        activations = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}
        activation = activations[config.get('activation', 'gelu')]
        if float(config.get('dropout', 0.)) != 0.:
            raise ValueError('This likelihood model fixes dropout to zero')
        self.logvar_min = float(config.get('logvar_min', -12.))
        self.logvar_max = float(config.get('logvar_max', 6.))
        self.beta = float(config.get('beta', 1.))
        self.tau = float(config.get('tau', 1.))
        self.rho = float(config.get('rho', 0.))
        self.rank_samples = int(config.get('rank_samples', 16))
        scalars = (self.logvar_min, self.logvar_max, self.beta, self.tau, self.rho)
        if (not all(math.isfinite(value) for value in scalars)
                or self.beta <= 0 or self.tau <= 0 or not 0 <= self.rho < 1
                or self.logvar_min >= self.logvar_max):
            raise ValueError('Invalid variance bounds or probability-model parameters')
        if self.rank_samples < 2 or self.rank_samples % 2:
            raise ValueError('rank_samples must be a positive even integer, at least two')

        def network(widths):
            layers = []
            for before, after in zip(widths[:-1], widths[1:]):
                layers.extend([nn.Linear(before, after), activation()])
            return layers

        self.encoder_net = nn.Sequential(*network([self.input_dim] + dims))
        self.to_mu = nn.Linear(dims[-1], 1)
        self.to_logvar = nn.Linear(dims[-1], 1)
        self.decoder_net = nn.Sequential(
            *network([1] + list(reversed(dims))), nn.Linear(dims[0], self.input_dim))
        torch.manual_seed(int(config.get('seed', 42)))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)

    def posterior(self, x):
        hidden = self.encoder_net(x)
        return (self.to_mu(hidden).squeeze(-1),
                self.to_logvar(hidden).squeeze(-1).clamp(self.logvar_min, self.logvar_max))

    def encode(self, x):
        return self.posterior(x)[0]

    def decode(self, z):
        if z.ndim == 1:
            z = z.unsqueeze(-1)
        return self.decoder_net(z)

    def forward(self, x):
        mu, logvar = self.posterior(x)
        z = mu + torch.exp(.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar


def _validate_mask(mask, width, device):
    mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if mask.ndim != 2 or mask.shape[0] < 1 or mask.shape[1] != width or width < 1:
        raise ValueError('mask must be a nonempty subject-by-maximum-visits matrix')
    if not bool(mask[:, 0].all()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError('Each subject must contain a nonempty contiguous valid prefix')
    return mask


def _subject_indices(padded_indices, mask, length, device):
    indices = torch.as_tensor(padded_indices, dtype=torch.long, device=device)
    if indices.ndim != 2:
        raise ValueError('padded_indices must be a subject-by-maximum-visits matrix')
    mask = _validate_mask(mask, indices.shape[1], device)
    if indices.shape != mask.shape:
        raise ValueError('padded_indices and mask must have identical shapes')
    valid = indices[mask]
    if bool(((valid < 0) | (valid >= length)).any()):
        raise ValueError('A valid subject index is outside the scan batch')
    return torch.where(mask, indices, 0), mask


def ascending_pl_nll(padded_z, mask, tau):
    """Full ascending-order NLL per subject, with optional leading draw axes.

    For visits 1,...,m this is sum_{r=2}^m
    log(1 + sum_{j<r} exp((z_j-z_r)/tau)). Padding follows all valid visits.
    Finite padding avoids an all-minus-infinity logcumsumexp and its NaN gradient.
    """
    if padded_z.ndim < 2 or not math.isfinite(float(tau)) or tau <= 0:
        raise ValueError('Expected subject/visit axes and a positive finite temperature')
    mask = _validate_mask(mask, padded_z.shape[-1], padded_z.device)
    if padded_z.shape[-2:] != mask.shape:
        raise ValueError('The final two latent axes must match the mask')
    safe_z = torch.where(mask, padded_z, 0.)
    centered = safe_z - safe_z[..., :1]
    scaled = torch.where(mask, centered / tau, 0.)
    prefixes = torch.logcumsumexp(scaled, dim=-1)
    # Excluding the current visit before softplus avoids subtracting two
    # nearly equal large logsumexp values when its order probability is near one.
    terms = F.softplus(prefixes[..., :-1] - scaled[..., 1:])
    return torch.where(mask[..., 1:], terms, 0.).sum(dim=-1)


def joint_gaussian_kl(mu, logvar, padded_indices, mask, rho):
    """KL(q_i || N(0, (1-rho)I+rho*11')) for each complete subject.

    q_i has independent visit marginals. The mean quadratic is evaluated in
    the common-direction/orthogonal decomposition to avoid cancellation.
    """
    if not math.isfinite(float(rho)) or not 0 <= rho < 1:
        raise ValueError('rho must satisfy 0 <= rho < 1')
    if mu.ndim != 1 or logvar.shape != mu.shape:
        raise ValueError('mu and logvar must be scan vectors of identical shape')
    indices, mask = _subject_indices(padded_indices, mask, len(mu), mu.device)
    means = torch.where(mask, mu[indices], 0.)
    logvars = torch.where(mask, logvar[indices], 0.)
    variances = torch.where(mask, logvars.exp(), 0.)
    m = mask.sum(dim=-1).to(mu.dtype)
    mean = means.sum(dim=-1) / m
    centered = torch.where(mask, means - mean[:, None], 0.)
    a = 1. - rho
    b = 1. + (m - 1.) * rho
    inverse_diagonal = (m - 1.) / (m * a) + 1. / (m * b)
    trace = inverse_diagonal * variances.sum(dim=-1)
    quadratic = centered.square().sum(dim=-1) / a + m * mean.square() / b
    logdet = (m - 1.) * math.log1p(-rho) + torch.log1p((m - 1.) * rho)
    return .5 * (trace + quadratic - m + logdet - logvars.sum(dim=-1))


def loss_terms(model, x, padded_indices, mask, normalizer):
    """Return rec, rank, kl; the objective is rec + beta * (rank + kl).

    For b uniformly sampled subjects among n training subjects, supply
    normalizer=b*N/n, where N is the full training scan count. This gives an
    unbiased estimate of the globally scan-normalized objective, even when
    subjects have different visit counts. Every scan in x must occur once.
    Reconstruction uses one draw; ranking uses rank_samples antithetic draws.
    """
    normalizer = float(normalizer)
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError('normalizer must be finite and positive')
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != model.input_dim:
        raise ValueError('x must be a nonempty scan-by-feature matrix')
    indices, mask = _subject_indices(padded_indices, mask, len(x), x.device)
    valid = indices[mask]
    if (valid.numel() != len(x)
            or not torch.equal(torch.sort(valid).values, torch.arange(len(x), device=x.device))):
        raise ValueError('Subject rows must partition the scan batch exactly once')
    mu, logvar = model.posterior(x)
    std = torch.exp(.5 * logvar)
    reconstruction_z = mu + std * torch.randn_like(mu)
    rec = (model.decode(reconstruction_z) - x).square().sum() / (x.shape[1] * normalizer)
    half_eps = torch.randn((model.rank_samples // 2, len(x)), dtype=mu.dtype, device=mu.device)
    eps = torch.cat((half_eps, -half_eps), dim=0)
    ranking_z = mu[None, :] + std[None, :] * eps
    rank = ascending_pl_nll(ranking_z[:, indices], mask, model.tau).sum(dim=-1).mean() / normalizer
    kl = joint_gaussian_kl(mu, logvar, indices, mask, model.rho).sum() / normalizer
    return {'rec': rec, 'rank': rank, 'kl': kl}


def score_model(model, xnp):
    """Return the deterministic single-PET variational expectation in float64."""
    x = np.asarray(xnp, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != model.input_dim or len(x) == 0:
        raise ValueError('xnp must be a nonempty scan-by-feature matrix')
    if not np.isfinite(x).all():
        raise FloatingPointError('Nonfinite scoring input')
    scoring = deepcopy(model).cpu().double().eval()
    with torch.no_grad():
        mu, logvar = scoring.posterior(torch.from_numpy(x))
        variance = logvar.exp()
    stage_mean = mu.numpy()
    stage_logvar = logvar.numpy()
    stage_variance = variance.numpy()
    if not (np.isfinite(stage_mean).all() and np.isfinite(stage_logvar).all()
            and np.isfinite(stage_variance).all() and np.all(stage_variance > 0)):
        raise FloatingPointError('Nonfinite or invalid encoder stage distribution')
    return dict(stage_mean=stage_mean, stage_variance=stage_variance,
                stage_logvar=stage_logvar,
                diagnostics=dict(score_definition='E_q[Z|x]=encoder_mu',
                                 uses_decoder=False, inference_dtype='float64',
                                 inference_mode='eval', single_scan_only=True))
