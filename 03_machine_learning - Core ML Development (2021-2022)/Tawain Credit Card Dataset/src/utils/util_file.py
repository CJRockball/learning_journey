import pandas as pd
from sklearn.utils import resample

# Use dict to translate
GENDER_dict = {1: "male", 2: "female"}
GENDER_inv = {v: k for k, v in GENDER_dict.items()}
EDUCATION_dict = {
    0: "education_other",
    1: "graduate_school",
    2: "university",
    3: "high_school",
    4: "education_other",
    5: "education_other",
    6: "education_other",
}
EDUCATION_inv = {v: k for k, v in EDUCATION_dict.items()}
MARRIAGE_dict = {0: "marriage_other", 1: "married", 2: "single", 3: "marriage_other"}
MARRIAGE_inv = {v: k for k, v in MARRIAGE_dict.items()}
label_dict = {0: "yes", 1: "no"}
label_inv = {v: k for k, v in label_dict.items()}


def upsample_data(X_train, y_train):
    """ Upsamples data

    Args:
        X_train ([type]): [description]
        X_test ([type]): [description]
        y_train ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Separate majority and minority classes
    index_majority = y_train[y_train["default.payment.next.month"] == 0].index.values
    index_minority = y_train[y_train["default.payment.next.month"] == 1].index.values

    df_majority = X_train.loc[index_majority, :]
    df_minority = X_train.loc[index_minority, :]

    print(
        "df_majority.shape", df_majority.shape, "df_minority.shape", df_minority.shape
    )

    # Upsample minority class
    df_minority_upsampled = resample(
        df_minority,
        replace=True,  # sample with replacement
        n_samples=df_majority.shape[0],  # to match majority class
        random_state=42,
    )  # reproducible results

    print(
        "df_majority.shape:",
        df_majority.shape,
        "df_minority_upsampled.shape:",
        df_minority_upsampled.shape,
    )

    # Combine majority class with upsampled minority class
    X_train_upsampled = pd.concat([df_majority, df_minority_upsampled])
    y_train_upsampled = y_train.loc[X_train_upsampled.index, :]

    X_train_upsampled = X_train_upsampled.reset_index(drop=True)
    y_train_upsampled = y_train_upsampled.reset_index(drop=True)

    # Display new class counts
    print("X_train_upsampled.shape:", X_train_upsampled.shape)
    print("y_train_upsampled.shape:", y_train_upsampled.shape)
    return X_train_upsampled, y_train_upsampled


def pre_pipeline_process(X, no_pay=True, bill=True):
    """[summary]

    Args:
        X ([type]): [description]
        no_pay (bool, optional): [description]. Defaults to True.
        bill (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    # Remove cols ID
    X = X.drop(columns=["ID"])

    if no_pay:
        # Remove cols ID and Pay_0 to Pay_7
        cols_to_drop = ["PAY_" + str(i) for i in range(7)]
        X = X.drop(columns=cols_to_drop)

    if bill:
        # Use BAL_AMT = BILL_AMT - PAY_AMT
        BAL_cols = ["BAL_AMT" + str(i + 1) for i in range(6)]
        BILL_cols = ["BILL_AMT" + str(i + 1) for i in range(6)]
        PAY_cols = ["PAY_AMT" + str(i + 1) for i in range(6)]

        for i, j, k in zip(BAL_cols, BILL_cols, PAY_cols):
            X[i] = X[j] - X[k]
            X = X.drop(columns=[j, k])

    # Combine categories in categorical data
    X["EDUCATION"] = X["EDUCATION"].replace(EDUCATION_dict)
    X["MARRIAGE"] = X["MARRIAGE"].replace(MARRIAGE_dict)
    X["SEX"] = X["SEX"].replace(GENDER_dict)
    return X
