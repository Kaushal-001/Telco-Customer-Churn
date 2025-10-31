import pandas as pd

#_ this in the beginning of the function name to indicate it's private 
# function and will be used only inside this module

# s is expected to be a pandas series like column of a dataframe
def _map_binary_series(s: pd.Series) -> pd.Series: 
    """
    Apply deterministic binary encoding to 2-category features.
    
    This function implements the core binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers. The mappings
    are deterministic and must be consistent between training and serving.

    """
    # Get unique values and remove NaN
    # below line of code is use to extract unique non-null values from the series s
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    if valset == {"Yes", "No"}:
        return s.map({"Yes": 1, "No": 0}).astype("Int64")  
    
    if valset == {"Male", "Female"}:
        return s.map({"Male": 1,"Female": 0}).astype("Int64")
    
    # ==== Generic binary mapping ====
    # For any other 2-category feature, use stable alphabetical ordering
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[1]: 1, sorted_vals[0]: 0}
        return s.map(mapping).astype("Int64")
    
     # === NON-BINARY FEATURES ===
    # Return unchanged - will be handled by one-hot encoding
    return s

def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data.
    
    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the
    serving pipeline to ensure prediction accuracy.

    """

    df = df.copy()
    print(f"🔧 Starting feature engineering on {df.shape[1]} columns...")

    # === STEP 1: Identify Feature Types ===
    # Find categorical columns (object dtype) excluding the target variable

    obj_cols = [col for col in df.select_dtypes(include=["object"]).columns if col!= target_col]
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    print(f"   📊 Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")

    # === STEP 2: split columns by cardinality(number of categories) ===
    # binary columns: get binary encoding
    # multi-category columns: one-hot encoding
    binary_cols = [c for c in obj_cols if df[c].nunique()==2]
    multi_cat_cols = [c for c in obj_cols if df[c].nunique()>2]

    print(f"   🔢 {len(binary_cols)} binary and {len(multi_cat_cols)} multi-category features identified")

    # === STEP 3: Apply Binary Encoding ===
    # convert binary categorical columns to 0/1 using deterministic mapping
    for col in binary_cols:
        original_dtype = df[col].dtype
        df[col] = _map_binary_series(df[col].astype(str))
        print(f"      - Binary encoded column '{col}' from {original_dtype} to {df[col].dtype}")

    
    # === STEP 4: Convert boolean columsn ===
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    if bool_cols:
        df[bool_cols] = df[bool_cols].astype("Int64")
        print(f"   ✅ Converted {len(bool_cols)} boolean columns to Int64")

    # === STEP 5: One-Hot Encoding for Multi-Category Features ===
    if multi_cat_cols:
        print(f"   🏷️ Applying one-hot encoding to {len(multi_cat_cols)} multi-category features...")
        original_shape = df.shape
        df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

        new_features = df.shape[1]-original_shape[1] + len(multi_cat_cols)
        print(f"      - One-hot encoding added {new_features} new features. New shape: {df.shape}")

    
    # === STEP 6: DATA TYPE CLEANING ===
    # Ensure all numeric columns are of type float64 or Int64
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            # fill na value with 0 before converting to Int64
            df[c] = df[c].fillna(0).astype("Int64")
    
    print(f"✅ Feature engineering complete: {df.shape[1]} final features")
    return df


