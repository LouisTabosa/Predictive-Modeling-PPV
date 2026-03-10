import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os
import itertools
import time

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
D_MAX = 2000.0
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2.6,
    "figure.dpi": 150,
    "axes.unicode_minus": True,
    "mathtext.default": "regular",
})

# True parameters (from synthetic data)
TRUE_PARAMS = {"c": 1200.0, "a": 0.8, "b": -1.6, "gamma": -0.005}

# Colors and Symbols — 4 models
COLORS = {"Empirical": "#F39C12", "ANN": "#3498DB", "EDML": "#38761d", "SVR": "#9B59B6"}
MARKERS = {"Empirical": "o", "ANN": "s", "EDML": "^", "SVR": "v"}

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================
def synth_ppv_dataset(N=1000, D_max=2000.0, seed=42, use_attenuation=True):
    """
    Scenario A (use_attenuation=False):
        PPV = C * Q^a * D^b * noise
    Scenario B (use_attenuation=True):
        PPV = C * Q^a * D^b * exp(gamma * D) * noise
    Log-normal multiplicative noise
    """
    rng_local = np.random.default_rng(seed)
    c = TRUE_PARAMS["c"]
    a = TRUE_PARAMS["a"]
    b = TRUE_PARAMS["b"]
    gamma = TRUE_PARAMS["gamma"]

    D = 10.0 ** rng_local.uniform(np.log10(50), np.log10(D_max), size=N)
    Q = 10.0 ** rng_local.uniform(np.log10(180), np.log10(1100), size=N)
    noise = np.exp(rng_local.normal(0, 1, size=N))  # log-normal

    PPV_true = c * (Q ** a) * (D ** b)
    if use_attenuation:
        PPV_true = PPV_true * np.exp(gamma * D)

    PPV_obs = PPV_true * noise

    return D, Q, PPV_obs, PPV_true

# ============================================================
# EMPIRICAL MODEL (baseline): log PPV = log C + a log Q + b log D
# ============================================================
def fit_empirical(D, Q, ylog):
    """Linear regression in log-log space (power-law)."""
    X = np.column_stack([np.log10(Q), np.log10(D)])
    model = LinearRegression().fit(X, ylog)
    return model

def predict_empirical(model, D, Q):
    X = np.column_stack([np.log10(Q), np.log10(D)])
    return model.predict(X)

def get_empirical_params(model):
    return 10 ** model.intercept_, model.coef_[0], model.coef_[1]

# ============================================================
# FEATURES PARA ANN E EDML
# ============================================================
def build_features_ann(D, Q):
    """Features to ANN: [D, Q]"""
    
    return np.column_stack([D, Q])

def build_features_edml(D, Q, ylog_emp, gamma=None):
    """
    Features to EDML ANN: [D, Q, SD, log D, log Q, ylog_emp]
    + [gamma * D] for Scenario B (when gamma != None)
    """
    SD = D / np.sqrt(Q)
    X = np.column_stack([D, Q, SD, np.log10(D), np.log10(Q), ylog_emp])
    if gamma is not None:
        X = np.column_stack([X, gamma * D])
    return X

# ============================================================
# GRID SEARCH PARA GAMMA (EDML)
# ============================================================
def search_gamma(Dtr, Qtr, rtr, yemp_tr, Dva, Qva, rva, yemp_va,
                 gamma_range=None, label="EDML-gamma"):
    """
    Grid search to determine the best gamma.
    For each candidate gamma, train a fixed ANN on the residual
    and evaluate on the validation set.
    Returns: best_gamma, df with grid search results.
    """
    if gamma_range is None:
        gamma_range = np.linspace(-0.01, 0.0, 201)  # step=0.00005, hits -0.00505 and -0.00495

    # ANN fixed for quick evaluation of gamma
    fixed_cfg = {
        "hidden_layer_sizes": (32, 16),
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "batch_size": 32,
        "activation": "relu",
    }

    results = []
    best_rmse = np.inf
    best_gamma = None

    for i, g in enumerate(gamma_range):
        Xtr_g = build_features_edml(Dtr, Qtr, yemp_tr, gamma=g)
        Xva_g = build_features_edml(Dva, Qva, yemp_va, gamma=g)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                solver="adam", max_iter=2000,
                early_stopping=True, validation_fraction=0.2,
                n_iter_no_change=25, random_state=RANDOM_STATE,
                **fixed_cfg
            ))
        ])
        pipe.fit(Xtr_g, rtr)
        rhat = pipe.predict(Xva_g)
        rmse_val = float(np.sqrt(mean_squared_error(rva, rhat)))
        r2_val = float(r2_score(rva, rhat))
        results.append({"gamma": g, "rmse": rmse_val, "r2": r2_val})

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_gamma = g

        if (i + 1) % max(1, len(gamma_range) // 5) == 0 or (i + 1) == len(gamma_range):
            print(f"    {label} trial {i+1}/{len(gamma_range)} | "
                  f"gamma={g:.5f} | RMSE={rmse_val:.6f} | best gamma={best_gamma:.5f}")

    df_gs = pd.DataFrame(results)
    print(f"    >>> Melhor gamma recuperado: {best_gamma:.6f} "
          f"(verdadeiro: {TRUE_PARAMS['gamma']})")
    return best_gamma, df_gs


# ============================================================
# TUNING ANN (generic — used for ANN and EDML)
# ============================================================
def tune_ann(Xtr, ytr, Xva, yva, label="ANN"):
    """
    Grid search for ANN.
    """
    hidden_pool = [(16,), (32,), (64,), (16, 16), (32, 16), (32, 32)]
    alpha_pool = [1e-5, 1e-4, 1e-3]
    lr_pool = [5e-4, 1e-3, 2e-3]
    batch_pool = [16, 32, 64, 128, 256]
    act_pool = ["relu", "tanh"]

    best_rmse = np.inf
    best_cfg = None
    best_model = None

    all_configs = list(itertools.product(hidden_pool, alpha_pool, lr_pool, batch_pool, act_pool))
    total = len(all_configs)

    for i, (hidden, alpha, lr, batch, act) in enumerate(all_configs):
        cfg = {
            "hidden_layer_sizes": hidden,
            "alpha": float(alpha),
            "learning_rate_init": float(lr),
            "batch_size": int(batch),
            "activation": act,
        }
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                solver="adam", max_iter=2000,
                early_stopping=True, validation_fraction=0.2,
                n_iter_no_change=25, random_state=RANDOM_STATE,
                **cfg
            ))
        ])
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xva)
        rmse_val = float(np.sqrt(mean_squared_error(yva, yhat)))

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_cfg = cfg
            best_model = pipe

        if (i + 1) % (max(1, total // 10)) == 0 or (i + 1) == total:
            print(f"    {label} Grid Search trial {i+1}/{total} | best RMSE: {best_rmse:.6f}")

    return best_model, best_cfg, best_rmse

# ============================================================
# TUNING SVR
# ============================================================
def tune_svr(Xtr, ytr, Xva, yva, label="SVR"):
    """
    Grid search for Support Vector Regression (SVR - RBF kernel).
    """
    C_pool = [1.0, 10.0, 50.0, 100.0]
    epsilon_pool = [0.01, 0.05, 0.1, 0.2]
    gamma_svr_pool = ["scale", "auto", 0.01, 0.1]

    best_rmse = np.inf
    best_cfg = None
    best_model = None

    all_configs = list(itertools.product(C_pool, epsilon_pool, gamma_svr_pool))
    total = len(all_configs)

    for i, (C, eps, g_svr) in enumerate(all_configs):
        cfg = {
            "C": float(C),
            "epsilon": float(eps),
            "gamma_svr": g_svr,
        }
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=float(C), epsilon=float(eps), gamma=g_svr))
        ])
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xva)
        rmse_val = float(np.sqrt(mean_squared_error(yva, yhat)))

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_cfg = cfg
            best_model = pipe

        if (i + 1) % (max(1, total // 5)) == 0 or (i + 1) == total:
            print(f"    {label} Grid Search trial {i+1}/{total} | best RMSE: {best_rmse:.6f}")

    return best_model, best_cfg, best_rmse


# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

# ============================================================
# VISUALIZATIONS
# ============================================================
def save_obs_vs_pred(y_obs, pred_dict, title, filename):
    """Observed vs Predicted for all models."""
    fig, ax = plt.subplots(figsize=(10, 8.5))
    for name, (yp, met) in pred_dict.items():
        c = COLORS.get(name, "gray")
        m = MARKERS.get(name, "o")
        label = f"{name} (RMSE={met['RMSE']:.4f}, R$^2$={met['R2']:.4f})"
        ax.scatter(y_obs, yp, s=60, alpha=0.6, color=c, marker=m,
                   label=label, edgecolors="k", linewidths=0.3)
    mn = min(y_obs.min(), min(yp.min() for yp, _ in pred_dict.values()))
    mx = max(y_obs.max(), max(yp.max() for yp, _ in pred_dict.values()))
    margin = (mx - mn) * 0.05
    ax.plot([mn - margin, mx + margin], [mn - margin, mx + margin],
            "k--", linewidth=3.0, label="1:1 line", zorder=0)
    ax.set_xlabel(r"Observed $\log_{10}(\mathrm{PPV})$")
    ax.set_ylabel(r"Predicted $\log_{10}(\mathrm{PPV})$")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {filename}")


def save_error_vs_distance(D, y_obs, pred_dict, title, filename):
    """Absolute relative error vs distance."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, (yp, _) in pred_dict.items():
        c = COLORS.get(name, "gray")
        m = MARKERS.get(name, "o")
        PPV_obs = 10 ** y_obs
        PPV_pred = 10 ** yp
        rel_err = np.abs(PPV_pred - PPV_obs) / PPV_obs
        ax.scatter(D, rel_err, s=40, alpha=0.5, color=c, marker=m,
                   label=name, edgecolors="k", linewidths=0.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance D (m)")
    ax.set_ylabel("Absolute relative error")
    ax.set_title(title)
    ax.legend(fontsize=13)
    ax.grid(True, which="both", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {filename}")


def save_ppv_vs_distance(D, PPV_obs, D_grid, Q_ref, curves, true_curve_ppv,
                         title, filename):
    """PPV vs Distance with model curves and true curve."""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(D, PPV_obs, s=15, alpha=0.25, color="gray",
               label="Observed data", zorder=1, edgecolors="k", linewidths=0.1)
    for name, ppv_curve, color, ls in curves:
        ax.plot(D_grid, ppv_curve, color=color, linewidth=2.5,
                linestyle=ls, label=name, zorder=3)
    ax.plot(D_grid, true_curve_ppv, "k-", linewidth=3, alpha=0.5,
            label="True model", zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Distance D (m)"); ax.set_ylabel("PPV")
    ax.set_title(title)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {filename}")


def save_residuals_boxplot(residuals_dict, title, filename):
    """Boxplot of residuals."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data, labels, colors_box = [], [], []
    for name, res in residuals_dict.items():
        data.append(res); labels.append(name)
        colors_box.append(COLORS.get(name, "gray"))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                    showfliers=True, flierprops=dict(marker="o", markersize=3, alpha=0.4),
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_ylabel(r"Residual $\log_{10}(\mathrm{PPV})$")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {filename}")


def save_params_comparison(params_true, params_recovered, filename):
    """Visual table of true vs recovered parameters."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    rows = []
    for key in ["c", "a", "b"]:
        vt = params_true[key]
        vr = params_recovered.get(key, "—")
        if isinstance(vr, float):
            err = abs(vr - vt) / abs(vt) * 100
            rows.append([key, f"{vt}", f"{vr:.4f}", f"{err:.2f}%"])
        else:
            rows.append([key, f"{vt}", str(vr), "—"])
    table = ax.table(cellText=rows,
                     colLabels=["Parameter", "True", "Recovered", "Error (%)"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.8)
    ax.set_title("True vs Recovered Parameters (Empirical)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {filename}")


def save_computational_cost(times_dict, title, filename):
    """Bar chart of computational cost (execution time) by model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(times_dict.keys())
    times = list(times_dict.values())
    colors_bar = [COLORS.get(n, "gray") for n in names]

    bars = ax.bar(names, times, color=colors_bar, edgecolor="k", linewidth=0.8,
                  alpha=0.85, width=0.5)

    # Labels on top of each bar
    for bar, t in zip(bars, times):
        if t < 1.0:
            label = f"{t*1000:.1f} ms"
        else:
            label = f"{t:.2f} s"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {filename}")

# ============================================================
# RUN SCENARIO
# ============================================================
def run_scenario(scenario_name, use_attenuation):
    """
    Executes a complete scenario with 3 models:
      1) Empirical (power-law in log-log)
      2) ANN (direct prediction)
      3) EDML (Empirical + ANN on residual)
    """
    print("\n" + "=" * 70)
    label = "com atenuacao (exp(-gamma*D))" if use_attenuation else "sem atenuacao"
    print(f"SCENARIO {scenario_name} — {label}")
    print("=" * 70)
    print(f"True parameters: {TRUE_PARAMS}")

    # ----------------------------------------------------------
    # 1) Gerar dados sintéticos
    # ----------------------------------------------------------
    print(f"\n[1] Generating synthetic data (Scenario {scenario_name})...")
    N = 1000
    D, Q, PPV_obs, PPV_true = synth_ppv_dataset(
        N=N, D_max=D_MAX, seed=42, use_attenuation=use_attenuation
    )
    ylog = np.log10(PPV_obs)
    print(f"    N={N}, D=[{np.min(D):.1f}, {np.max(D):.1f}] m, "
          f"Q=[{np.min(Q):.1f}, {np.max(Q):.1f}] kg")

    df_data = pd.DataFrame({"D": D, "Q": Q, "PPV_obs": PPV_obs, "PPV_true": PPV_true})
    df_data.to_csv(f"{OUT_DIR}/Scenario{scenario_name}_synthetic_data.csv", index=False)

    # ----------------------------------------------------------
    # 2) Split train / validation / test
    # ----------------------------------------------------------
    print("\n[2] Split: 60% train, 20% validation, 20% test")
    idx = np.arange(N)
    tr_idx, te_idx = train_test_split(idx, test_size=0.20, random_state=RANDOM_STATE)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.25, random_state=RANDOM_STATE)

    Dtr, Qtr, ytr = D[tr_idx], Q[tr_idx], ylog[tr_idx]
    Dva, Qva, yva = D[va_idx], Q[va_idx], ylog[va_idx]
    Dte, Qte, yte = D[te_idx], Q[te_idx], ylog[te_idx]
    print(f"    Train: {len(tr_idx)}, Validation: {len(va_idx)}, Test: {len(te_idx)}")

    # ----------------------------------------------------------
    # 3) EMPIRICAL — power-law regression
    # ----------------------------------------------------------
    print("\n[3] Modelo Empirical: log PPV = log C + a log Q + b log D")
    t0_emp = time.perf_counter()
    emp = fit_empirical(Dtr, Qtr, ytr)
    yemp_te = predict_empirical(emp, Dte, Qte)
    t_emp = time.perf_counter() - t0_emp
    met_emp = compute_metrics(yte, yemp_te)
    c_e, a_e, b_e = get_empirical_params(emp)
    print(f"    c={c_e:.4f}, a={a_e:.4f}, b={b_e:.4f}")
    print(f"    RMSE={met_emp['RMSE']:.4f}, R2={met_emp['R2']:.4f}")
    print(f"    Tempo: {t_emp:.4f} s")

    # ----------------------------------------------------------
    # 4) ANN — direct prediction of log(PPV)
    # ----------------------------------------------------------
    print(f"\n[4] Modelo ANN (direct prediction): features = [D, Q]")
    t0_ann = time.perf_counter()
    Xtr_ann = build_features_ann(Dtr, Qtr)
    Xva_ann = build_features_ann(Dva, Qva)
    Xte_ann = build_features_ann(Dte, Qte)

    best_ann, ann_cfg, _ = tune_ann(Xtr_ann, ytr, Xva_ann, yva, label="ANN")

    #   Retrain on train + validation
    print("    Retraining ANN on train+validation...")
    Xtrva_ann = np.vstack([Xtr_ann, Xva_ann])
    ytrva = np.concatenate([ytr, yva])

    final_ann = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            solver="adam", max_iter=2000,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, random_state=RANDOM_STATE,
            **ann_cfg
        ))
    ])
    final_ann.fit(Xtrva_ann, ytrva)

    yann_te = final_ann.predict(Xte_ann)
    t_ann = time.perf_counter() - t0_ann
    met_ann = compute_metrics(yte, yann_te)
    print(f"    Best ANN config: {ann_cfg}")
    print(f"    RMSE={met_ann['RMSE']:.4f}, R2={met_ann['R2']:.4f}")
    print(f"    Tempo: {t_ann:.2f} s")

    # ----------------------------------------------------------
    # 4.5) SVR — Support Vector Regression (RBF)
    # ----------------------------------------------------------
    print(f"\n[4.5] Modelo SVR (RBF kernel, direct prediction): features = [D, Q]")
    t0_svr = time.perf_counter()
    best_svr, svr_cfg, _ = tune_svr(Xtr_ann, ytr, Xva_ann, yva, label="SVR")

    #   Retrain on train + validation
    print("    Retraining SVR on train+validation...")
    final_svr = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=svr_cfg["C"], epsilon=svr_cfg["epsilon"], gamma=svr_cfg["gamma_svr"]))
    ])
    final_svr.fit(Xtrva_ann, ytrva)

    ysvr_te = final_svr.predict(Xte_ann)
    t_svr = time.perf_counter() - t0_svr
    met_svr = compute_metrics(yte, ysvr_te)
    print(f"    Best SVR config: {svr_cfg}")
    print(f"    RMSE={met_svr['RMSE']:.4f}, R2={met_svr['R2']:.4f}")
    print(f"    Tempo: {t_svr:.2f} s")


    # ----------------------------------------------------------
    # 5) EDML — Empirical + ANN on residual
    # ----------------------------------------------------------
    print(f"\n[5] Modelo EDML: Empirical + ANN on residual")
    t0_edml = time.perf_counter()

    yemp_tr = predict_empirical(emp, Dtr, Qtr)
    yemp_va = predict_empirical(emp, Dva, Qva)

    rtr = ytr - yemp_tr
    rva = yva - yemp_va

    # 5a) Grid search for gamma (Scenario B) or None (Scenario A)
    if use_attenuation:
        print("    [5a] Grid search for gamma...")
        best_gamma, df_gs = search_gamma(
            Dtr, Qtr, rtr, yemp_tr, Dva, Qva, rva, yemp_va
        )
        gamma_feat = best_gamma
        # Save grid search results
        df_gs.to_csv(f"{OUT_DIR}/Scenario{scenario_name}_gamma_grid_search.csv", index=False)
    else:
        gamma_feat = None
        best_gamma = None
        df_gs = None

    feat_desc = "[D, Q, SD, logD, logQ, ylog_emp, gamma*D]" if use_attenuation else "[D, Q, SD, logD, logQ, ylog_emp]"
    print(f"    Features: {feat_desc}")

    # 5b) Tuning ANN on residual with best gamma
    Xtr_edml = build_features_edml(Dtr, Qtr, yemp_tr, gamma=gamma_feat)
    Xva_edml = build_features_edml(Dva, Qva, yemp_va, gamma=gamma_feat)
    Xte_edml = build_features_edml(Dte, Qte, yemp_te, gamma=gamma_feat)

    best_edml, edml_cfg, _ = tune_ann(Xtr_edml, rtr, Xva_edml, rva, label="EDML")

    #   Retrain on train + validation
    print("    Retraining EDML ANN on train+validation...")
    Xtrva_edml = np.vstack([Xtr_edml, Xva_edml])
    rtrva = np.concatenate([rtr, rva])

    final_edml = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            solver="adam", max_iter=2000,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, random_state=RANDOM_STATE,
            **edml_cfg
        ))
    ])
    final_edml.fit(Xtrva_edml, rtrva)

    rhat_te = final_edml.predict(Xte_edml)
    yedml_te = yemp_te + rhat_te
    t_edml = time.perf_counter() - t0_edml
    met_edml = compute_metrics(yte, yedml_te)
    print(f"    Best EDML config: {edml_cfg}")
    if best_gamma is not None:
        print(f"    Gamma recovered: {best_gamma:.6f} (true: {TRUE_PARAMS['gamma']})")
    print(f"    RMSE={met_edml['RMSE']:.4f}, R2={met_edml['R2']:.4f}")
    print(f"    Tempo: {t_edml:.2f} s")

    # ----------------------------------------------------------
    # 6) Comparative table (metrics + computational cost)
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Comparative table — Scenario {scenario_name} (test set, log10 space)")
    print("=" * 70)

    rows_table = [
        {"Modelo": "Empirical", **met_emp, "Tempo_s": round(t_emp, 4)},
        {"Modelo": "ANN", **met_ann, "Tempo_s": round(t_ann, 2)},
        {"Modelo": "SVR", **met_svr, "Tempo_s": round(t_svr, 2)},
        {"Modelo": "EDML", **met_edml, "Tempo_s": round(t_edml, 2)},
    ]
    df_table = pd.DataFrame(rows_table)
    print(df_table.to_string(index=False))
    df_table.to_csv(f"{OUT_DIR}/Scenario{scenario_name}_comparative_table.csv", index=False)

    # ----------------------------------------------------------
    # 7) Plots
    # ----------------------------------------------------------
    print(f"\n[7] Generating plots (Scenario {scenario_name})...")
    prefix = f"Scenario{scenario_name}"

    pred_dict = {
        "Empirical": (yemp_te, met_emp),
        "ANN": (yann_te, met_ann),
        "SVR": (ysvr_te, met_svr),
        "EDML": (yedml_te, met_edml),
    }

    # 7a) Observed vs Predicted
    save_obs_vs_pred(yte, pred_dict,
                     f"Observed vs Predicted — Scenario {scenario_name}",
                     f"{OUT_DIR}/{prefix}_fig_obs_vs_pred.png")

    # 7b) Error vs Distance
    save_error_vs_distance(Dte, yte, pred_dict,
                           f"Error vs Distance — Scenario {scenario_name}",
                           f"{OUT_DIR}/{prefix}_fig_error_vs_distance.png")

    # 7c) PPV vs Distance
    Q_ref = np.median(Q)
    D_grid = np.linspace(np.min(D), D_MAX, 500)

    # Predictions of the 3 models on the grid
    yemp_grid = predict_empirical(emp, D_grid, np.full_like(D_grid, Q_ref))
    Xgrid_ann = build_features_ann(D_grid, np.full_like(D_grid, Q_ref))
    yann_grid = final_ann.predict(Xgrid_ann)
    ysvr_grid = final_svr.predict(Xgrid_ann)
    yemp_grid_for_edml = yemp_grid  # empirical prediction as feature
    Xgrid_edml = build_features_edml(D_grid, np.full_like(D_grid, Q_ref),
                                     yemp_grid_for_edml, gamma=gamma_feat)
    rhat_grid = final_edml.predict(Xgrid_edml)
    yedml_grid = yemp_grid + rhat_grid

    # True curve
    ppv_true_grid = TRUE_PARAMS["c"] * (Q_ref ** TRUE_PARAMS["a"]) * \
                    (D_grid ** TRUE_PARAMS["b"])
    if use_attenuation:
        ppv_true_grid = ppv_true_grid * np.exp(TRUE_PARAMS["gamma"] * D_grid)

    curves = [
        ("Empirical", 10 ** yemp_grid, COLORS["Empirical"], "--"),
        ("ANN", 10 ** yann_grid, COLORS["ANN"], ":"),
        ("SVR", 10 ** ysvr_grid, COLORS["SVR"], "-"),
        ("EDML", 10 ** yedml_grid, COLORS["EDML"], "-."),
    ]
    save_ppv_vs_distance(D, PPV_obs, D_grid, Q_ref, curves, ppv_true_grid,
                         f"PPV vs Distance (Q = {Q_ref:.0f} kg) — Scenario {scenario_name}",
                         f"{OUT_DIR}/{prefix}_fig_ppv_vs_distance.png")

    # 7d) Boxplot of residuals
    save_residuals_boxplot(
        {"Empirical": yte - yemp_te, "ANN": yte - yann_te, "SVR": yte - ysvr_te, "EDML": yte - yedml_te},
        f"Residuals distribution — Scenario {scenario_name}",
        f"{OUT_DIR}/{prefix}_fig_residuals_boxplot.png"
    )

    # 7e) Computational cost — bar chart
    save_computational_cost(
        {"Empirical": t_emp, "ANN": t_ann, "SVR": t_svr, "EDML": t_edml},
        f"Computational Cost — Scenario {scenario_name}",
        f"{OUT_DIR}/{prefix}_fig_computational_cost.png"
    )

    # ----------------------------------------------------------
    # 8) Summary
    # ----------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Summary — Scenario {scenario_name}")
    print(f"{'=' * 70}")
    print(f"\nModels:"
          f"\nEmpirical:  (RMSE={met_emp['RMSE']:.4f}, R2={met_emp['R2']:.4f}, Tempo={t_emp:.4f}s)"
          f"\nANN:        (RMSE={met_ann['RMSE']:.4f}, R2={met_ann['R2']:.4f}, Tempo={t_ann:.2f}s)"
          f"\nSVR:        (RMSE={met_svr['RMSE']:.4f}, R2={met_svr['R2']:.4f}, Tempo={t_svr:.2f}s)"
          f"\nEDML:       (RMSE={met_edml['RMSE']:.4f}, R2={met_edml['R2']:.4f}, Tempo={t_edml:.2f}s)")

    print(f"\nFiles in {OUT_DIR}/:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith(prefix):
            print(f"  - {f}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("PPV ATTENUATION — TWO SCENARIOS × THREE MODELS")
    print("=" * 70)
    print(f"True parameters: {TRUE_PARAMS}")

    # Scenario A — without attenuation (baseline)
    run_scenario("A", use_attenuation=False)

    # Scenario B — with material attenuation
    run_scenario("B", use_attenuation=True)

    print("\n\n" + "=" * 70)
    print("COMPLETE EXECUTION — All files generated in:")
    print(f"  {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()