import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

# Set the working directory.
base_dir = "/BDproject1/User/liugaojing/Project/Monkey-MNPs/Year2024/Analysis9/Organ-Sensitive/ConfirmFunction4"
os.makedirs(base_dir, exist_ok=True)
os.chdir(base_dir)

# Read the cardiovascular sn data.
CardiovascularAdataClean = sc.read_h5ad("/BDproject1/User/liugaojing/Project/Monkey-MNPs/Year2024/Analysis9/CardiovascularAdataClean.harmony.final.15.h5ad")


# Remove the specified organs (requirement: filter out the ovary, spleen, and thymus).

adata = CardiovascularAdataClean[~CardiovascularAdataClean.obs["Organ"].isin(["Ovary", "Spleen", "Thymus"])].copy()


adata.obs["Monkey_ID"] = (
    adata.obs["Donor"].astype(str) + "_" +
    adata.obs["Condition"].astype(str)
)
monkey_col = "Monkey_ID"


# Prepare the global dense matrix and labels.

X_full = adata.layers["log1p"].toarray()
y = adata.obs["Condition"].astype("category").cat.codes
idx = np.arange(adata.n_obs)


# 60:20:20 cell-level stratified split.

X_tr, X_tmp, y_tr, y_tmp, idx_tr, idx_tmp = train_test_split(
    X_full, y, idx, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
    X_tmp, y_tmp, idx_tmp, test_size=0.5, stratify=y_tmp, random_state=42)


# Feature engineering aligned with the hold-out method (HVG filtering + Top 50 feature mining).

adata_tr = adata[idx_tr].copy()


adata_tr.X = adata_tr.layers["log1p"].copy()

sc.pp.highly_variable_genes(adata_tr, batch_key=monkey_col, n_top_genes=3000, flavor="seurat")
hvg_indices = np.where(adata_tr.var["highly_variable"])[0]

X_tr_hvg = X_tr[:, hvg_indices]
rf0 = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
).fit(X_tr_hvg, y_tr)

k_small = 50
importances = rf0.feature_importances_
top_hvg_relative_idx = np.argsort(importances)[-k_small:]
top_idx = hvg_indices[top_hvg_relative_idx]

gene_names = adata.var_names.values
feat_imp_all = pd.DataFrame({"gene": gene_names[hvg_indices], "importance": importances}).sort_values("importance", ascending=False)
feat_imp_all.to_csv("RF_feature_importance_all_hvg_genes.txt", sep="\t", index=False)

feat_imp_top = pd.DataFrame({"gene": gene_names[top_idx], "importance": importances[top_hvg_relative_idx]})
feat_imp_top.sort_values("importance", ascending=False).to_csv("top_50_genes_names.txt", sep="\t", index=False)
np.save("top_50_genes_indices.npy", top_idx)

# Train the final model with a minimal feature set (Top 50).

X_tr_sel = X_tr[:, top_idx]
X_val_sel = X_val[:, top_idx]
X_test_sel = X_test[:, top_idx]

rf = RandomForestClassifier(
    n_estimators=300,
    max_features=0.5,
    class_weight="balanced",
    oob_score=True,
    n_jobs=-1,
    random_state=42
).fit(X_tr_sel, y_tr)

# Save the model object.
model_filename = 'final_random_forest_model.joblib'
joblib.dump(rf, model_filename)



# Model evaluation and whole-cell probability export.


val_pred = rf.predict(X_val_sel)
val_prob = rf.predict_proba(X_val_sel)[:, 1]
print(f"\nVal AUROC = {roc_auc_score(y_val, val_prob):.4f}")
print(classification_report(y_val, val_pred, target_names=["Blank", "NPs"]))


test_prob = rf.predict_proba(X_test_sel)[:, 1]
test_pred = rf.predict(X_test_sel)
print(f"Test AUROC = {roc_auc_score(y_test, test_prob):.4f}")
print(classification_report(y_test, test_pred, target_names=["Blank", "NPs"]))


pd.Series(adata.obs_names[idx_test]).to_csv("test_set_cells.txt", index=False, header=False)


set_array = np.full(adata.n_obs, "unused", dtype=object)
set_array[idx_tr] = "train"
set_array[idx_val] = "val"
set_array[idx_test] = "test"
adata.obs["set"] = set_array


X_full_sel = X_full[:, top_idx]
prob_all = rf.predict_proba(X_full_sel)
adata.obs["prob_blank_all"] = prob_all[:, 0]
adata.obs["prob_nPS_all"] = prob_all[:, 1]
adata.obs[["prob_blank_all", "prob_nPS_all", "set"]].to_csv("All_cells_predicted_probabilities.txt", sep="\t")

# Calculate organ sensitivity based on test set cells.

test_obs = adata.obs.iloc[idx_test].copy()
test_obs["prob_nPS"] = test_prob

mean_prob = test_obs.groupby(["Organ", "Condition"], observed=True)["prob_nPS"].mean().unstack(fill_value=np.nan)

organ_sensitivity = (mean_prob["NPs"] - mean_prob["Blank"]).dropna().sort_values(ascending=False).rename("delta_prob")

print("\n=== Organ-level sensitivity (Δ mean P(nPS) on Test Set) ===")
print(organ_sensitivity)
organ_sensitivity.to_csv("All.organ.sensitive.txt", sep="\t")

fpr_test, tpr_test, _ = roc_curve(y_test, test_prob)
fig1 = plt.figure(figsize=(5, 4))
plt.plot(fpr_test, tpr_test, label=f"ROC curve (AUC = {roc_auc_score(y_test, test_prob):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Random-Forest ROC – Test set")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("All.organ.ROC.svg")
plt.close(fig1)

cm_test = confusion_matrix(y_test, test_pred, labels=[0, 1])
fig2, ax2 = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["Blank", "NPs"]).plot(ax=ax2, colorbar=False)
plt.title("Confusion matrix – Test set")
plt.tight_layout()
plt.savefig("RF.confusion.matrix.svg")
plt.close(fig2)

fpr_val, tpr_val, _ = roc_curve(y_val, val_prob)
fig3 = plt.figure(figsize=(5, 4))
plt.plot(fpr_val, tpr_val, label=f"ROC curve (AUC = {roc_auc_score(y_val, val_prob):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Random-Forest ROC – Validation set")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("Validation.ROC.svg")
plt.close(fig3)

cm_val = confusion_matrix(y_val, val_pred, labels=[0, 1])
fig4, ax4 = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=["Blank", "NPs"]).plot(ax=ax4, colorbar=False)
plt.title("Confusion matrix – Validation set")
plt.tight_layout()
plt.savefig("RF.validation.confusion.matrix.svg")
plt.close(fig4)

fig5 = plt.figure(figsize=(6, max(4, len(organ_sensitivity) * 0.35)))
organ_sensitivity.plot.barh(color="crimson", edgecolor="black")
plt.xlabel("Mean Δ P(NPs)")
plt.ylabel("Organ")
plt.title("Organ sensitivity ranking (higher = more sensitive)")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Organ.sensitive.svg")
plt.close(fig5)
