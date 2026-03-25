import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Full alpha grid
# -----------------------------
alphas_all = np.array(
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    dtype=float
)

# -----------------------------
# Set alpha_max HERE
# -----------------------------
alpha_max = 1.0
alpha_mask = alphas_all <= alpha_max
alphas = alphas_all[alpha_mask]

# -----------------------------
# Output file
# -----------------------------
output_file = "TAUC/sciq_qwen_results.csv"

# -----------------------------
# UA / CA values — SciQ
# -----------------------------
UA_all = {
    "KWT(LLM)":   [70.4, 71.1, 71.8, 72.6, 73.3, 74.0, 74.7, 75.4, 76.2, 76.9, 77.6],
    "KWT(Rouge)": [63.2, 64.5, 65.9, 67.2, 68.5, 69.9, 71.2, 72.5, 73.8, 75.2, 76.5],
    "KWT(EM)":    [56.9, 58.6, 60.3, 61.9, 63.6, 65.3, 67.0, 68.7, 70.3, 72.0, 73.7],
    "R-Tuning":   [49.7, 51.7, 53.7, 55.7, 57.7, 59.8, 61.8, 63.8, 65.8, 67.8, 69.8],
    "SEAL":       [70.7, 71.0, 71.4, 71.7, 72.0, 72.4, 72.7, 73.0, 73.3, 73.7, 74.0],
    "KWT-RF":     [63.4, 64.7, 66.1, 67.4, 68.8, 70.1, 71.4, 72.8, 74.1, 75.5, 76.8],
    "KWT-U":      [66.3, 67.5, 68.7, 69.9, 71.1, 72.3, 73.4, 74.6, 75.8, 77.0, 78.2],
}

CA_all = {
    "KWT(LLM)":   [70.4, 70.2, 70.0, 69.8, 69.6, 69.4, 69.2, 69.0, 68.8, 68.6, 68.4],
    "KWT(Rouge)": [63.2, 62.4, 61.7, 60.9, 60.2, 59.4, 58.6, 57.9, 57.1, 56.4, 55.6],
    "KWT(EM)":    [56.9, 55.5, 54.2, 52.8, 51.4, 50.1, 48.7, 47.3, 45.9, 44.6, 43.2],
    "R-Tuning":   [49.7, 47.3, 44.9, 42.6, 40.2, 37.8, 35.4, 33.0, 30.7, 28.3, 25.9],
    "SEAL":       [70.7] * 11,
    "KWT-RF":     [63.4, 62.8, 62.2, 61.6, 61.0, 60.4, 59.8, 59.2, 58.6, 58.0, 57.4],
    "KWT-U":      [66.3, 65.9, 65.4, 65.0, 64.5, 64.1, 63.6, 63.2, 62.7, 62.3, 61.8],
}



UA = {k: np.array(v, dtype=float)[alpha_mask] for k, v in UA_all.items()}
CA = {k: np.array(v, dtype=float)[alpha_mask] for k, v in CA_all.items()}

# -----------------------------
# (1) Your current nTAUC: AUC of CA w.r.t UA (average CA over UA-span)
#     + robustify by sorting along UA
# -----------------------------
def ntauc_ca_over_ua(ua_vals, ca_vals):
    ua = np.asarray(ua_vals, dtype=float)
    ca = np.asarray(ca_vals, dtype=float)

    # Sort by UA to ensure a proper "area under CA(UA)" interpretation
    order = np.argsort(ua)
    ua_s = ua[order]
    ca_s = ca[order]

    d_ua = np.diff(ua_s)
    auc = np.sum(0.5 * (ca_s[:-1] + ca_s[1:]) * np.abs(d_ua))

    ua_range = ua_s.max() - ua_s.min()
    n_auc = auc / ua_range if ua_range > 0 else np.nan
    return auc, n_auc, ua_range

# -----------------------------
# (2) Joint-AUC: integrates BOTH UA and CA over alpha
#     - This is closer to "trade-off summary" in the sense:
#       high only when BOTH UA and CA are high.
#     - Normalize by 100 because your values are percentages.
#     - Use trapezoid rule over alpha.
# -----------------------------
def joint_auc_over_alpha(alphas, ua_vals, ca_vals, alpha_max):
    ua = np.asarray(ua_vals, dtype=float) 
    ca = np.asarray(ca_vals, dtype=float) 
    a  = np.asarray(alphas, dtype=float)

    score = ua * ca / 100  # joint desirability
    auc = np.trapz(score, a)  # integral over alpha
    n_auc = auc / alpha_max if alpha_max > 0 else np.nan
    return auc, n_auc

# -----------------------------
# Compute results
# -----------------------------
rows = []
for model in UA:
    tauc, n_tauc, ua_range = ntauc_ca_over_ua(UA[model], CA[model])
    jauc, n_jauc = joint_auc_over_alpha(alphas, UA[model], CA[model], alpha_max)

    rows.append({
        "Dataset": "SciQ",
        "Backbone": "Qwen",
        "Model": model,
        "alpha_max": alpha_max,
        "TAUC_CA_over_UA": tauc,
        "UA_range": ua_range,
        "nTAUC_CA_over_UA": n_tauc,     # (기존 정의) 평균 CA(UA)
        "JointAUC_alpha": jauc,
        "nJointAUC_alpha": n_jauc,      # (의도 보강) UA와 CA를 함께 반영
    })

df = pd.DataFrame(rows)

# 두 가지 기준 모두 보고 싶으면 둘 다 정렬해서 확인
df_ntauc = df.sort_values("nTAUC_CA_over_UA", ascending=False)
df_joint = df.sort_values("nJointAUC_alpha", ascending=False)

print("\n=== Sorted by nTAUC (CA over UA) ===")
print(df_ntauc.to_string(index=False))

print("\n=== Sorted by nJointAUC (UA*CA over alpha) ===")
print(df_joint.to_string(index=False))

# -----------------------------
# Save
# -----------------------------
if output_file:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

# -----------------------------
# Plot UA–CA curves
# -----------------------------
plt.figure(figsize=(7, 5))
for model in UA:
    plt.plot(UA[model], CA[model], marker="o", label=model)

plt.xlabel("UA-Acc(α)")
plt.ylabel("CA-Acc(α)")
plt.title(f"SciQ: UA–CA Trade-off Curves (α ∈ [0, {alpha_max}])")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Plot score vs alpha (optional): UA, CA, and joint UA*CA
# -----------------------------
plt.figure(figsize=(7, 5))
for model in UA:
    joint = (UA[model] / 100.0) * (CA[model] / 100.0)
    plt.plot(alphas, joint, marker="o", label=model)

plt.xlabel("α")
plt.ylabel("UA(α) × CA(α) (normalized)")
plt.title(f"SciQ: Joint desirability over α (α ∈ [0, {alpha_max}])")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
