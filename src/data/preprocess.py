import pandas as pd

def preprocess_data(df : pd.DataFrame, target_col: str ="Churn") -> pd.DataFrame:
    """
    Preprocess the input dataframe :
    1. trim columns name
    2. drop obvious id cols
    3. fix totalcharges to numeric
    4. map target churn to 0/1 if needed
    5. simple NA handling
    """
    df.columns = df.columns.str.strip()

    #drop id if presents
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])


    #target to 0/1 if its YES/NO
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"Yes":1, "No":0})

    #totalcharges often has blanks in this dataset ->coerce to float
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    #senior citizen should be 0/1 if presents
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype("int")

    #simple NA handling
    # -numeirc cols -> fillna with 0
    # - others: leave for encoders to handle (get_dummies ignores NaN safely)

    num_cols = df.select_dtypes(include=["number"]).columns #number includes both int64 and float64
    df[num_cols] = df[num_cols].fillna(0)

    return df 
    