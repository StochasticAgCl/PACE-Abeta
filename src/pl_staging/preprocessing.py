"""CN-reference covariate adjustment for regional PET measurements."""
import numpy as np

def _mean_impute_covs(covs, means=None):
    covs = np.asarray(covs, dtype=float).copy()
    n_cov = covs.shape[1]

    if means is None:
        means = np.full(n_cov, np.nan)
        for j in range(n_cov):
            col = covs[:, j]
            if np.all(np.isnan(col)):
                raise ValueError(f"Covariate column {j} is all NaN, cannot do mean imputation.")
            means[j] = np.nanmean(col)

    for j in range(n_cov):
        nan_idx = np.isnan(covs[:, j])
        covs[nan_idx, j] = means[j]

    return covs, means


def fit_preadjust_params_simple(pheno, covs, dx, dx_code):
    pheno = np.asarray(pheno, dtype=float)
    covs = np.asarray(covs, dtype=float)
    dx = np.asarray(dx)

    n_sbj, n_res = pheno.shape
    n_cov = covs.shape[1]

    adj_pheno = np.full(pheno.shape, np.nan)
    betas = np.full((n_res, n_cov + 1), np.nan)   # intercept + covs
    means = np.full((n_res, n_cov), np.nan)

    if dx_code < 0:
        # fit imputation means using all subjects
        covs_imp, cov_impute_means = _mean_impute_covs(covs)

        covs_new = np.column_stack((np.ones(n_sbj), covs_imp))

        for i_res in range(n_res):
            resp = pheno[:, i_res]
            non_miss = np.where(~np.isnan(resp))[0]

            if len(non_miss) > 0:
                beta = np.linalg.lstsq(covs_new[non_miss, :], resp[non_miss], rcond=None)[0]
                betas[i_res, :] = beta.ravel()

                B = beta[1:].ravel()
                M = np.mean(covs_imp[non_miss, :], axis=0)
                means[i_res, :] = M

                # apply adjustment to all subjects with non-missing phenotype
                for j_idx in non_miss:
                    adj_pheno[j_idx, i_res] = pheno[j_idx, i_res] - np.sum(B * (covs_imp[j_idx, :] - M))

    else:
        # fit only on subset dx == dx_code
        sbj_idx = np.where(dx == dx_code)[0]
        if len(sbj_idx) == 0:
            return adj_pheno, betas, means

        covs_fit = covs[sbj_idx, :]
        pheno_fit = pheno[sbj_idx, :]

        # imputation mean estimated from fitting subset only
        covs_fit_imp, cov_impute_means = _mean_impute_covs(covs_fit)

        # apply the same imputation means to all subjects
        covs_all_imp, _ = _mean_impute_covs(covs, means=cov_impute_means)

        covs_new = np.column_stack((np.ones(len(sbj_idx)), covs_fit_imp))

        for i_res in range(n_res):
            resp_fit = pheno_fit[:, i_res]
            non_miss_fit = np.where(~np.isnan(resp_fit))[0]

            resp_all = pheno[:, i_res]
            non_miss_all = np.where(~np.isnan(resp_all))[0]

            if len(non_miss_fit) > 0:
                beta = np.linalg.lstsq(covs_new[non_miss_fit, :], resp_fit[non_miss_fit], rcond=None)[0]
                betas[i_res, :] = beta.ravel()

                B = beta[1:].ravel()
                M = np.mean(covs_fit_imp[non_miss_fit, :], axis=0)
                means[i_res, :] = M

                # apply adjustment to all subjects with non-missing phenotype
                for j_idx in non_miss_all:
                    adj_pheno[j_idx, i_res] = pheno[j_idx, i_res] - np.sum(B * (covs_all_imp[j_idx, :] - M))

    return adj_pheno, betas, means


def apply_preadjust_params_simple(pheno, covs, betas, means, cov_impute_means=None):
    pheno = np.asarray(pheno, dtype=float)
    covs = np.asarray(covs, dtype=float)
    betas = np.asarray(betas, dtype=float)
    means = np.asarray(means, dtype=float)

    n_sbj, n_res = pheno.shape
    n_cov = covs.shape[1]
    adj = np.full(pheno.shape, np.nan)

    # if not provided, use means from current covs
    covs_imp, used_impute_means = _mean_impute_covs(covs, means=cov_impute_means)

    for i_res in range(n_res):
        if np.all(np.isnan(betas[i_res, :])):
            continue

        B = betas[i_res, 1:].ravel()
        M = means[i_res, :].ravel()

        non_miss = np.where(~np.isnan(pheno[:, i_res]))[0]

        for j_idx in non_miss:
            adj[j_idx, i_res] = pheno[j_idx, i_res] - np.sum(B * (covs_imp[j_idx, :] - M))

    return adj