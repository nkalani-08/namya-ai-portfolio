import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler


# ══════════════════════════════════════════════════════════════════
#  MISSING VALUES
# ══════════════════════════════════════════════════════════════════

def recommend_missing_values(df):
    """Return per-column imputation recommendations."""
    recommendations = {}

    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue

        missing_percent = (missing_count / len(df)) * 100
        dtype = df[col].dtype

        if missing_percent > 50:
            recommendations[col] = {
                "Recommendation": "Drop Column",
                "Alternatives": ["Arbitrary Value Imputation"],
                "Reason": f"{missing_percent:.1f}% values are missing — too little information to impute reliably."
            }

        elif dtype in ["int64", "float64"]:
            skew = abs(df[col].dropna().skew())
            if skew < 0.5:
                rec, reason = "Mean Imputation", f"Skewness = {skew:.2f} (approx. normal). Mean is a good centre estimate."
            else:
                rec, reason = "Median Imputation", f"Skewness = {skew:.2f} (skewed). Median is robust to extreme values."

            recommendations[col] = {
                "Recommendation": rec,
                "Alternatives": ["Mean Imputation", "Median Imputation", "KNN Imputation", "CCA", "Arbitrary Value Imputation"],
                "Reason": reason
            }

        else:
            recommendations[col] = {
                "Recommendation": "Mode Imputation",
                "Alternatives": ["CCA", "Arbitrary Category Imputation"],
                "Reason": f"{missing_percent:.1f}% values missing. Mode preserves the most frequent category."
            }

    return recommendations


def mean_imputation(df, col):
    df = df.copy()
    df[col] = df[col].fillna(df[col].mean())
    return df

def median_imputation(df, col):
    df = df.copy()
    df[col] = df[col].fillna(df[col].median())
    return df

def mode_imputation(df, col):
    df = df.copy()
    mode_val = df[col].mode()
    if len(mode_val) > 0:
        df[col] = df[col].fillna(mode_val[0])
    return df

def drop_column(df, col):
    return df.drop(columns=[col])

def cca(df, col):
    return df.dropna(subset=[col])

def arbitrary_value(df, col, value=-999):
    df = df.copy()
    df[col] = df[col].fillna(value)
    return df

def arbitrary_category(df, col):
    df = df.copy()
    df[col] = df[col].fillna("Missing")
    return df

def knn_imputation(df, cols, n_neighbors=5):
    df = df.copy()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[cols] = imputer.fit_transform(df[cols])
    return df

def apply_imputation(df, col, method):
    if method == "Mean Imputation":       return mean_imputation(df, col)
    if method == "Median Imputation":     return median_imputation(df, col)
    if method == "Mode Imputation":       return mode_imputation(df, col)
    if method == "CCA":                   return cca(df, col)
    if method == "Arbitrary Value Imputation":    return arbitrary_value(df, col)
    if method == "Arbitrary Category Imputation": return arbitrary_category(df, col)
    return df


# ══════════════════════════════════════════════════════════════════
#  DUPLICATES
# ══════════════════════════════════════════════════════════════════

def recommend_duplicates(df):
    dup_count = df.duplicated().sum()
    if dup_count == 0:
        return {"Recommendation": "No Action Needed", "Reason": "No duplicate rows found."}
    return {"Recommendation": "Remove Duplicates", "Reason": f"{dup_count} duplicate rows detected."}

def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
#  OUTLIERS
# ══════════════════════════════════════════════════════════════════

def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return int(((df[col] < lower) | (df[col] > upper)).sum())

def recommend_outliers(df, col):
    outlier_count = detect_outliers_iqr(df, col)
    if outlier_count == 0:
        return {"Recommendation": "No Treatment Needed", "Reason": "No outliers detected."}

    skew = abs(df[col].dropna().skew())
    if skew < 0.5:
        rec    = "Z-Score Method"
        reason = f"{outlier_count} outliers. Skewness = {skew:.2f} (approx. normal) — Z-Score is appropriate."
    else:
        rec    = "IQR Method"
        reason = f"{outlier_count} outliers. Skewness = {skew:.2f} (skewed) — IQR is more robust."

    return {
        "Recommendation": rec,
        "Alternatives": ["IQR Method", "Z-Score Method"],
        "Reason": reason
    }

def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)].reset_index(drop=True)

def remove_outliers_zscore(df, col, threshold=3):
    temp = df[df[col].notnull()].copy()
    scores = np.abs(zscore(temp[col]))
    # Keep rows within threshold AND rows where col was null (already handled above)
    mask = scores < threshold
    return temp[mask].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
#  ENCODING
# ══════════════════════════════════════════════════════════════════

def recommend_encoding(df):
    recommendations = {}
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

    for col in categorical_cols:
        unique_values = df[col].nunique()

        if unique_values == 2:
            recommendations[col] = {
                "Recommendation": "Label Encoding",
                "Alternatives": ["One-Hot Encoding"],
                "Reason": f"Binary categorical feature ({unique_values} unique values)."
            }
        elif unique_values <= 10:
            recommendations[col] = {
                "Recommendation": "One-Hot Encoding",
                "Alternatives": ["Label Encoding", "Frequency Encoding"],
                "Reason": f"Low-cardinality feature ({unique_values} unique values)."
            }
        else:
            recommendations[col] = {
                "Recommendation": "Frequency Encoding",
                "Alternatives": ["Label Encoding"],
                "Reason": f"High-cardinality feature ({unique_values} unique values) — OHE would create too many columns."
            }

    return recommendations

def one_hot_encode(df, col):
    return pd.get_dummies(df, columns=[col], drop_first=True)

def label_encode(df, col):
    df = df.copy()
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    return df

def frequency_encode(df, col):
    df = df.copy()
    freq = df[col].value_counts()
    df[col] = df[col].map(freq)
    return df


# ══════════════════════════════════════════════════════════════════
#  SCALING
# ══════════════════════════════════════════════════════════════════

def recommend_scaling(df):
    """Only recommends scaling for numeric columns present in df right now."""
    recommendations = {}
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numerical_cols:
        outliers = detect_outliers_iqr(df, col)
        skew = abs(df[col].dropna().skew())

        if outliers > 0:
            rec    = "RobustScaler"
            reason = f"{outliers} outliers detected — RobustScaler is resistant to them."
        elif skew < 0.5:
            rec    = "StandardScaler"
            reason = f"Skewness = {skew:.2f} (approx. normal) — StandardScaler is ideal."
        else:
            rec    = "MinMaxScaler"
            reason = f"Skewness = {skew:.2f} (not normal) — MinMaxScaler bounds values to [0,1]."

        recommendations[col] = {
            "Recommendation": rec,
            "Alternatives": ["StandardScaler", "MinMaxScaler", "RobustScaler"],
            "Reason": reason
        }

    return recommendations

def standard_scale(df, col):
    df = df.copy()
    df[[col]] = StandardScaler().fit_transform(df[[col]])
    return df

def minmax_scale(df, col):
    df = df.copy()
    df[[col]] = MinMaxScaler().fit_transform(df[[col]])
    return df

def robust_scale(df, col):
    df = df.copy()
    df[[col]] = RobustScaler().fit_transform(df[[col]])
    return df


# ══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE  —  returns (cleaned_df, report_dict)
#  report_dict keys match exactly what index.html expects:
#    original_shape, cleaned_shape, missing_total, steps,
#    numeric_columns, categorical_columns, dropped_columns,
#    missing, duplicates, outliers, encoding, scaling
# ══════════════════════════════════════════════════════════════════

def preprocess_dataset(df):
    df_processed = df.copy()
    steps = []

    original_shape = df.shape
    dropped_columns = []
    missing_fixed = 0

    # ── 1. Missing Values ──────────────────────────────────────
    missing_rec = recommend_missing_values(df_processed)
    missing_report = {}

    for col, rec in missing_rec.items():
        method = rec["Recommendation"]
        missing_fixed += int(df_processed[col].isnull().sum())

        if method == "Drop Column":
            df_processed = drop_column(df_processed, col)
            dropped_columns.append(col)
            steps.append(f"Dropped column '{col}' — {rec['Reason']}")
        elif method == "Mean Imputation":
            df_processed = mean_imputation(df_processed, col)
            steps.append(f"'{col}': filled missing values with mean — {rec['Reason']}")
        elif method == "Median Imputation":
            df_processed = median_imputation(df_processed, col)
            steps.append(f"'{col}': filled missing values with median — {rec['Reason']}")
        elif method == "Mode Imputation":
            df_processed = mode_imputation(df_processed, col)
            steps.append(f"'{col}': filled missing values with mode — {rec['Reason']}")

        missing_report[col] = {"method": method, "reason": rec["Reason"]}

    # ── 2. Duplicates ──────────────────────────────────────────
    dup_rec = recommend_duplicates(df_processed)
    dup_count = int(df_processed.duplicated().sum())

    if dup_count > 0:
        df_processed = remove_duplicates(df_processed)
        steps.append(f"Removed {dup_count} duplicate row(s).")

    # ── 3. Outliers ────────────────────────────────────────────
    outlier_report = {}
    # Only treat continuous columns (more than 10 unique values)
    numerical_cols_before_enc = df_processed.select_dtypes(include=["int64", "float64"]).columns
    outlier_cols = [c for c in numerical_cols_before_enc if df_processed[c].nunique() > 10]

    for col in outlier_cols:
        rec = recommend_outliers(df_processed, col)
        method = rec["Recommendation"]

        if method == "IQR Method":
            df_processed = remove_outliers_iqr(df_processed, col)
            steps.append(f"'{col}': removed outliers using IQR — {rec['Reason']}")
        elif method == "Z-Score Method":
            df_processed = remove_outliers_zscore(df_processed, col)
            steps.append(f"'{col}': removed outliers using Z-Score — {rec['Reason']}")
        else:
            steps.append(f"'{col}': no outlier treatment needed.")

        outlier_report[col] = {"method": method, "reason": rec["Reason"]}

    # ── 4. Encoding ────────────────────────────────────────────
    encoding_rec = recommend_encoding(df_processed)
    encoding_report = {}

    for col, rec in encoding_rec.items():
        method = rec["Recommendation"]

        if method == "Label Encoding":
            df_processed = label_encode(df_processed, col)
        elif method == "One-Hot Encoding":
            df_processed = one_hot_encode(df_processed, col)
        elif method == "Frequency Encoding":
            df_processed = frequency_encode(df_processed, col)

        steps.append(f"'{col}': applied {method} — {rec['Reason']}")
        encoding_report[col] = {"method": method, "reason": rec["Reason"]}

    # ── 5. Scaling ─────────────────────────────────────────────
    # Build scaling recs AFTER encoding so column list is current
    scaling_rec = recommend_scaling(df_processed)
    scaling_report = {}

    for col, rec in scaling_rec.items():
        method = rec["Recommendation"]

        if method == "StandardScaler":
            df_processed = standard_scale(df_processed, col)
        elif method == "MinMaxScaler":
            df_processed = minmax_scale(df_processed, col)
        elif method == "RobustScaler":
            df_processed = robust_scale(df_processed, col)

        steps.append(f"'{col}': scaled with {method} — {rec['Reason']}")
        scaling_report[col] = {"method": method, "reason": rec["Reason"]}

    # ── Build report dict (matches index.html template exactly) ──
    report = {
        "original_shape":      list(original_shape),
        "cleaned_shape":       list(df_processed.shape),
        "missing_total":       missing_fixed,
        "steps":               steps,
        "numeric_columns":     df_processed.select_dtypes(include=["int64", "float64"]).columns.tolist(),
        "categorical_columns": df_processed.select_dtypes(include=["object", "category"]).columns.tolist(),
        "dropped_columns":     dropped_columns,
        "missing":             missing_report,
        "duplicates":          {"count": dup_count, "method": dup_rec["Recommendation"]},
        "outliers":            outlier_report,
        "encoding":            encoding_report,
        "scaling":             scaling_report,
    }

    return df_processed, report
