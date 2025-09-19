import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import LabelEncoder


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def preprocess(df: pd.DataFrame, target_col="doenca_cardiaca"):
    # Separar features e label
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Transformar labels (strings) em números
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, le
