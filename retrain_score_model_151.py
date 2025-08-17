from pathlib import Path
import time, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import sklearn

print("sklearn:", sklearn.__version__)

BASE = Path(__file__).resolve().parent
APP  = BASE / "CricketAIWebApp"

DATA = APP / "data" / "t20s_combined.csv.gz"
if not DATA.exists():
    DATA = APP / "data" / "t20s_combined.csv"
assert DATA.exists(), f"Dataset not found in {DATA.parent}"

t0 = time.time()
df = pd.read_csv(DATA)
need = {"match_id","start_date","runs_off_bat","striker","batting_team","bowling_team","venue"}
miss = need - set(df.columns)
if miss: raise ValueError(f"Missing columns: {miss}")
df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
df = df[df["start_date"].dt.year >= 2015].copy()

per_match = (df.groupby(['striker','batting_team','bowling_team','venue','match_id','start_date'])['runs_off_bat']
               .sum().reset_index(name='total_runs')
               .sort_values(['striker','start_date']))

WINDOW = 5
per_match['batter_form'] = (per_match.groupby('striker')['total_runs']
                            .transform(lambda s: s.shift(1).rolling(window=WINDOW, min_periods=1).mean())
                           ).fillna(0.0)

features = ['striker','batting_team','bowling_team','venue','batter_form']
X = pd.get_dummies(per_match[features])
y = per_match['total_runs'].astype(float)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=180, max_depth=16, min_samples_leaf=2, n_jobs=-1, random_state=42)
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print("MAE:", round(mean_absolute_error(yte, pred), 2), "R^2:", round(r2_score(yte, pred), 2))

OUT = APP / "player_score_model"
OUT.mkdir(parents=True, exist_ok=True)
joblib.dump(model, OUT / "score_model.pkl")
joblib.dump(list(X.columns), OUT / "feature_names.pkl")
print("Saved:", OUT / "score_model.pkl")
print("Saved:", OUT / "feature_names.pkl")
