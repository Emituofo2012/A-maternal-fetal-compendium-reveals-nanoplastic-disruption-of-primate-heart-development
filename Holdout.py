import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import joblib
import itertools
import os

base_dir = "/BDproject1/User/liugaojing/Project/Monkey-MNPs/Year2024/Analysis9/Organ-Sensitive/ConfirmFunction3"
os.makedirs(base_dir, exist_ok=True)
os.chdir(base_dir)

adata = sc.read_h5ad("/BDproject1/User/liugaojing/Project/Monkey-MNPs/Year2024/Analysis9/Organ-Sensitive/ConfirmFunction/adata.no.Ovary.Spleen.Thymus.h5ad")

X_full = adata.layers["log1p"].toarray()
y_series = adata.obs["Condition"].astype("category")
y = y_series.cat.codes


adata.obs["Monkey_ID"] = (
    adata.obs["Donor"].astype(str) + "_" +
    adata.obs["Condition"].astype(str)
)

monkey_col = "Monkey_ID" 
monkey_ids = adata.obs[monkey_col].unique().tolist()

blank_monkeys = sorted([m for m in monkey_ids if "Blank" in m])
nps_monkeys = sorted([m for m in monkey_ids if "NPs" in m])

print(f"\nBlank monkeys: {blank_monkeys}")
print(f"NPs monkeys: {nps_monkeys}")


holdout_pairs = list(itertools.product(blank_monkeys, nps_monkeys))

gene_names = adata.var_names.values


overall_summary = []

for holdout_idx, (blank_holdout, nps_holdout) in enumerate(holdout_pairs, start=1):
    
    combo_name = f"Combo{holdout_idx}_holdout_{blank_holdout}_{nps_holdout}"
    combo_dir = os.path.join(base_dir, combo_name)
    os.makedirs(combo_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"{combo_name}")
    print(f"留出: {blank_holdout} + {nps_holdout}")
    print(f"{'='*70}")
    
   
    holdout_mask = adata.obs[monkey_col].isin([blank_holdout, nps_holdout]).values
    remain_mask = ~holdout_mask
    
    remain_idx = np.where(remain_mask)[0]
    holdout_idx_arr = np.where(holdout_mask)[0]
    
    X_remain = X_full[remain_idx]
    y_remain = y.iloc[remain_idx].values
    
    X_holdout = X_full[holdout_idx_arr]
    y_holdout = y.iloc[holdout_idx_arr].values
    
    
    X_tr, X_val, y_tr, y_val, idx_tr_local, idx_val_local = train_test_split(
        X_remain, y_remain, np.arange(len(remain_idx)),
        test_size=0.25, stratify=y_remain, random_state=42
    )
    
   
    train_global_idx = remain_idx[idx_tr_local]
    val_global_idx = remain_idx[idx_val_local]
    test_global_idx = holdout_idx_arr
    

    pd.Series(adata.obs_names[train_global_idx]).to_csv(
        os.path.join(combo_dir, "train_cells.txt"), index=False, header=False
    )
    pd.Series(adata.obs_names[val_global_idx]).to_csv(
        os.path.join(combo_dir, "val_cells.txt"), index=False, header=False
    )
    pd.Series(adata.obs_names[test_global_idx]).to_csv(
        os.path.join(combo_dir, "test_cells.txt"), index=False, header=False
    )
    
    
    adata_tr = adata[train_global_idx].copy()
    sc.pp.highly_variable_genes(adata_tr, batch_key=monkey_col, n_top_genes=2000, flavor="seurat")
    hvg_indices = np.where(adata_tr.var["highly_variable"])[0] 
    
    
    X_tr_hvg = X_tr[:, hvg_indices]
    rf0 = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42).fit(X_tr_hvg, y_tr)
    
    
    k_small = 50 
    top_hvg_relative_idx = np.argsort(rf0.feature_importances_)[-k_small:]
    top_idx = hvg_indices[top_hvg_relative_idx] 
    

    feat_imp_df = pd.DataFrame({"gene": gene_names[top_idx], "importance": rf0.feature_importances_[top_hvg_relative_idx]})
    feat_imp_df.sort_values("importance", ascending=False).to_csv(os.path.join(combo_dir, f"RF_gold_top_{k_small}_genes.txt"), sep="\t", index=False)

    X_tr_sel = X_tr[:, top_idx]
    X_val_sel = X_val[:, top_idx]
    X_test_sel = X_holdout[:, top_idx]
    
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", oob_score=True, n_jobs=-1, random_state=42).fit(X_tr_sel, y_tr)
    joblib.dump(rf, os.path.join(combo_dir, "final_random_forest_model.joblib"))
    
    val_prob = rf.predict_proba(X_val_sel)[:, 1]
    val_pred = rf.predict(X_val_sel)
    val_auc = roc_auc_score(y_val, val_prob)
    
    test_prob = rf.predict_proba(X_test_sel)[:, 1]
    test_pred = rf.predict(X_test_sel)
    test_auc = roc_auc_score(y_holdout, test_prob)
    
    fpr_val, tpr_val, _ = roc_curve(y_val, val_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr_val, tpr_val, label=f"AUC = {val_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Validation ROC – {combo_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(combo_dir, "Validation_ROC.svg"))
    plt.close()
    

    fpr_test, tpr_test, _ = roc_curve(y_holdout, test_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr_test, tpr_test, label=f"AUC = {test_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Holdout Test ROC – {combo_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(combo_dir, "Holdout_Test_ROC.svg"))
    plt.close()
    

    cm_val = confusion_matrix(y_val, val_pred, labels=[0, 1])
    disp_val = ConfusionMatrixDisplay(
        confusion_matrix=cm_val,
        display_labels=["Blank", "NPs"]
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    disp_val.plot(ax=ax, colorbar=False)
    plt.title(f"Validation Confusion Matrix – {combo_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(combo_dir, "Validation_Confusion_Matrix.svg"))
    plt.close()
    
    cm_test = confusion_matrix(y_holdout, test_pred, labels=[0, 1])
    disp_test = ConfusionMatrixDisplay(
        confusion_matrix=cm_test,
        display_labels=["Blank", "NPs"]
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    disp_test.plot(ax=ax, colorbar=False)
    plt.title(f"Holdout Test Confusion Matrix – {combo_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(combo_dir, "Holdout_Test_Confusion_Matrix.svg"))
    plt.close()
    
    X_full_sel = X_full[:, top_idx]
    prob_all = rf.predict_proba(X_full_sel)
    
    set_array = np.full(adata.n_obs, "unused", dtype=object)
    set_array[train_global_idx] = "train"
    set_array[val_global_idx] = "val"
    set_array[test_global_idx] = "test"
    
    prob_df = pd.DataFrame({
        "cell_id": adata.obs_names,
        "Organ": adata.obs["Organ"].values,
        "Condition": adata.obs["Condition"].values,
        "Monkey_ID": adata.obs[monkey_col].values,
        "prob_Blank": prob_all[:, 0],
        "prob_NPs": prob_all[:, 1],
        "set": set_array  
    })
    
    prob_df.to_csv(
        os.path.join(combo_dir, "All_cells_predicted_probabilities.txt"),
        sep="\t", index=False
    )
    
    
    test_obs_df = adata.obs.iloc[test_global_idx].copy()
    test_obs_df["prob_NPs"] = test_prob
    
    mean_prob = (
        test_obs_df
        .groupby(["Organ", "Condition"], observed=True)["prob_NPs"]
        .mean()
        .unstack(fill_value=np.nan)
    )
    
    cond_cols = mean_prob.columns.tolist()
    print(f"\nCondition: {cond_cols}")
    
    if "NPs" in cond_cols and "Blank" in cond_cols:
        organ_sensitivity = (
            mean_prob["NPs"] - mean_prob["Blank"]
        ).dropna().sort_values(ascending=False).rename("delta_prob")
        
        organ_sensitivity.to_csv(
            os.path.join(combo_dir, "Organ_sensitivity.txt"), sep="\t"
        )
        
        plt.figure(figsize=(6, max(4, len(organ_sensitivity) * 0.3)))
        organ_sensitivity.plot.barh()
        plt.xlabel("Mean Δ P(NPs)")
        plt.title(f"Organ sensitivity – {combo_name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(combo_dir, "Organ_sensitivity.svg"))
        plt.close()
        
        print(organ_sensitivity)
    else:
        organ_sensitivity = None
    

    overall_summary.append({
        "combo": combo_name,
        "holdout_blank": blank_holdout,
        "holdout_nps": nps_holdout,
        "n_train": len(train_global_idx),
        "n_val": len(val_global_idx),
        "n_test": len(test_global_idx),
        "val_auc": val_auc,
        "test_auc": test_auc,
        "oob_score": rf.oob_score_,
    })

summary_df = pd.DataFrame(overall_summary)
summary_df.to_csv(os.path.join(base_dir, "All_combos_summary.txt"), sep="\t", index=False)
