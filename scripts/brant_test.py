"""Brant-Test (1990, Separate-Fits-Ansatz) für die Proportional-Odds-Annahme.

Extrahiert aus 04_Regression.ipynb (H2), damit dieselbe, dort bereits validierte Logik in
06_Amazon_Robustness.ipynb (H3-Robustheitsprüfung) wiederverwendet werden kann, ohne Code zu
duplizieren oder das dortige finale Notebook zu verändern.

Nutzung:
    from brant_test import fit_threshold_logits, brant_test
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2


def fit_threshold_logits(X_with_const, y, thresholds):
    """Fit ein binäres Logit-Modell je kumulativer Schwelle (Rating > j)."""
    results, pi_hats = [], []
    for j in thresholds:
        y_bin = (y > j).astype(int)
        m = sm.Logit(y_bin, X_with_const).fit(disp=False, maxiter=200)
        results.append(m)
        pi_hats.append(m.predict(X_with_const))
    return results, pi_hats


def stacked_covariance(X_with_const, logit_results, pi_hats):
    """Blockmatrix der Kovarianzen der gestapelten Schwellen-Koeffizienten (Brant 1990)."""
    Xmat = X_with_const.values
    p_full = Xmat.shape[1]
    n_thresh = len(logit_results)

    W = [pi.values * (1 - pi.values) for pi in pi_hats]
    XtWjX_inv = [np.linalg.inv(Xmat.T @ (w[:, None] * Xmat)) for w in W]

    V = np.zeros((n_thresh * p_full, n_thresh * p_full))
    for j in range(n_thresh):
        V[j * p_full:(j + 1) * p_full, j * p_full:(j + 1) * p_full] = XtWjX_inv[j]
        for l in range(j + 1, n_thresh):
            Wjl = pi_hats[l].values - pi_hats[j].values * pi_hats[l].values
            XtWjlX = Xmat.T @ (Wjl[:, None] * Xmat)
            block = XtWjX_inv[j] @ XtWjlX @ XtWjX_inv[l]
            V[j * p_full:(j + 1) * p_full, l * p_full:(l + 1) * p_full] = block
            V[l * p_full:(l + 1) * p_full, j * p_full:(j + 1) * p_full] = block.T
    return V


def contrast_matrix(p_full, n_thresh, slope_idx):
    """D-Matrix: Differenzen der Steigungskoeffizienten zwischen aufeinanderfolgenden Schwellen."""
    n_gaps = n_thresh - 1
    D = np.zeros((n_gaps * len(slope_idx), n_thresh * p_full))
    row = 0
    for k in slope_idx:
        for m in range(n_gaps):
            D[row, m * p_full + k] = 1
            D[row, (m + 1) * p_full + k] = -1
            row += 1
    return D, n_gaps


def _wald_from_contrast(D_sub, beta_stack, V):
    """Wald-Statistik + p-Wert für eine beliebige Kontrast-Teilmatrix D_sub."""
    Db = D_sub @ beta_stack
    DVDt = D_sub @ V @ D_sub.T
    stat = float(Db @ np.linalg.solve(DVDt, Db))
    df = D_sub.shape[0]
    p = float(1 - chi2.cdf(stat, df))
    return stat, df, p


def brant_test(X_with_const, y, thresholds, model_label, groups=None):
    """Brant-Test (Wald, Separate-Fits) – Omnibus + je Prädiktor + optionale Variablen-Gruppen.

    `groups`: dict {Bezeichnung: [Spaltennamen]} für gemeinsame Blocktests mehrerer Prädiktoren.
    Ein solcher Gruppentest ist invariant gegenüber linearen Reparametrisierungen innerhalb der
    Gruppe, im Gegensatz zum Test einer einzelnen Spalte.
    """
    logit_results, pi_hats = fit_threshold_logits(X_with_const, y, thresholds)
    beta_stack = np.concatenate([m.params.values for m in logit_results])
    V = stacked_covariance(X_with_const, logit_results, pi_hats)

    p_full = X_with_const.shape[1]
    n_thresh = len(thresholds)
    slope_names = list(X_with_const.columns[1:])  # ohne Konstante
    slope_idx = list(range(1, p_full))

    D, n_gaps = contrast_matrix(p_full, n_thresh, slope_idx)
    wald_omnibus, df_omnibus, p_omnibus = _wald_from_contrast(D, beta_stack, V)

    rows = [{
        "Modell": model_label, "Test": "Brant (Wald)", "Variable": "Omnibus (alle Prädiktoren)",
        "Statistik": wald_omnibus, "df": df_omnibus, "p_Wert": p_omnibus,
    }]

    for i, name in enumerate(slope_names):
        Dk = D[i * n_gaps:(i + 1) * n_gaps, :]
        wald_k, df_k, p_k = _wald_from_contrast(Dk, beta_stack, V)
        rows.append({
            "Modell": model_label, "Test": "Brant (Wald)", "Variable": name,
            "Statistik": wald_k, "df": df_k, "p_Wert": p_k,
        })

    for group_label, group_cols in (groups or {}).items():
        rel_positions = [slope_names.index(c) for c in group_cols]
        row_idx = np.concatenate([np.arange(i * n_gaps, (i + 1) * n_gaps) for i in rel_positions])
        D_group = D[row_idx, :]
        wald_g, df_g, p_g = _wald_from_contrast(D_group, beta_stack, V)
        rows.append({
            "Modell": model_label, "Test": "Brant (Wald, gemeinsamer Block)", "Variable": group_label,
            "Statistik": wald_g, "df": df_g, "p_Wert": p_g,
        })

    return pd.DataFrame(rows)
