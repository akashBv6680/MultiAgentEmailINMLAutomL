from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

def run_manual_ml(df):
    df = df.copy().dropna()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    score = r2_score(y_test, model.predict(X_test))
    report = f"RandomForest processed {len(df)} rows. Features used: {list(X.columns)}"
    return report, score
