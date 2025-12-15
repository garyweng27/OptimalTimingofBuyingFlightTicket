import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# ==================================== Load clean dataset ====================================
df = pd.read_csv("./dataset/clean_dataset.csv")
df = df[df["class"] == "Economy"].reset_index(drop=True)
df["price"] = df["price"].astype(float)
df["duration_hr"] = df["duration"].astype(float)
stop_map = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3
}
df["stop_count"] = df["stops"].map(stop_map).fillna(2).astype(int)
df["route"] = df["source_city"] + "_" + df["destination_city"]


# =============================== Encode categorical variables ===============================
cat_cols = [
    "airline",
    "flight",
    "source_city",
    "destination_city",
    "departure_time",
    "arrival_time",
    "route"
]

encoders = {}
for col in cat_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col].astype(str))

# ==================================== Feature selection ====================================
feature_cols = [
    "airline",
    "flight",
    "route",
    "departure_time",
    "arrival_time",
    "duration_hr",
    "stop_count",
    "days_left"
]

X = df[feature_cols]
# y = df["price"]
y = np.log1p(df["price"]) 


# ================================ Train-test split (time-aware) ================================
df_sorted = df.sort_values("days_left")

X_train, X_test, y_train, y_test = train_test_split(
    df_sorted[feature_cols],
    np.log1p(df_sorted["price"]),
    # df_sorted["price"],
    test_size=0.2,
    shuffle=False
)

# ===================================== Train XGBoost model =====================================
model = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=11,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    eval_metric="rmse"
)

model.fit(X_train, y_train)

# ====================================== Evaluation ======================================
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

rmse = mean_squared_error(y_test_original, y_pred, squared=False)
mae = mean_absolute_error(y_test_original, y_pred)

print("RMSE (original scale):", rmse)
print("MAE (original scale):", mae)

# ============================= Model-based expected minimum price =============================
def predict_expected_min_price(model, base_features, max_days_left):
    rows = []
    for d in range(max_days_left, -1, -1):
        r = base_features.copy()
        r["days_left"] = d
        rows.append(r)

    X_future = pd.DataFrame(rows)
    log_preds = model.predict(X_future)
    price_preds = np.expm1(log_preds)

    best_idx = np.argmin(price_preds)

    curve_df = pd.DataFrame({
        "days_left": [r["days_left"] for r in rows],
        "predicted_price": price_preds
    })

    return (
        int(curve_df.iloc[best_idx]["days_left"]),
        float(curve_df.iloc[best_idx]["predicted_price"]),
        curve_df
    )

def evaluate_policy(df, model, feature_cols, min_days=20, agg_price="mean"):
    results = []
    grouped = df.groupby(["airline", "flight", "route"], sort=False)

    for key, g in grouped:
        if agg_price == "mean":
            price_curve = g.groupby("days_left")["price"].mean()
        elif agg_price == "min":
            price_curve = g.groupby("days_left")["price"].min()
        else:
            raise ValueError("agg_price must be 'mean' or 'min'")

        if price_curve.index.nunique() < min_days:
            continue

        max_day = int(price_curve.index.max())
        g_max = g[g["days_left"] == max_day].iloc[0]

        base = {
            "airline": g_max["airline"],
            "flight": g_max["flight"],
            "route": g_max["route"],
            "departure_time": g_max["departure_time"],
            "arrival_time": g_max["arrival_time"],
            "duration_hr": g_max["duration_hr"],
            "stop_count": g_max["stop_count"],
        }

        rows = []
        for d in range(max_day, -1, -1):
            r = base.copy()
            r["days_left"] = d
            rows.append(r)

        X_scan = pd.DataFrame(rows)[feature_cols]
        pred_prices = np.expm1(model.predict(X_scan))
        d_pred = int(rows[int(np.argmin(pred_prices))]["days_left"])

        d_true = int(price_curve.idxmin())
        p_true_min = float(price_curve.min())
        if d_pred not in price_curve.index:
            continue

        p_true_pred = float(price_curve.loc[d_pred])

        d_now = max_day
        p_now = float(price_curve.loc[d_now])

        results.append({
            "airline": key[0],
            "flight": key[1],
            "route": key[2],
            "max_day": max_day,
            "d_pred": d_pred,
            "d_true": d_true,
            "day_error": abs(d_pred - d_true),
            "regret": p_true_pred - p_true_min,
            "saving_vs_now": p_now - p_true_pred
        })

    return pd.DataFrame(results)

# ========================== Example usage (SAFE: sample from dataset) ==========================
if __name__ == "__main__":

    sample = df.sample(1, random_state=42).iloc[0]

    base_features = {
        "airline": sample["airline"],
        "flight": sample["flight"],
        "route": sample["route"],
        "departure_time": sample["departure_time"],
        "arrival_time": sample["arrival_time"],
        "duration_hr": sample["duration_hr"],
        "stop_count": sample["stop_count"]
    }

    current_days_left = int(sample["days_left"])

    best_day, best_price, curve = predict_expected_min_price(
        model,
        base_features,
        max_days_left=current_days_left
    )

    print("\n==============================")
    print("Model-based Expected Minimum")
    print("==============================")
    print(f"Sample route encoded id: {base_features['route']}")
    print(f"Current days_left: {current_days_left}")
    print(f"Best expected day to buy: {best_day}")
    print(f"Expected lowest price: {best_price:.2f}")
    dups = (
        df.groupby(["airline","flight","route","days_left"])
        .size()
        .reset_index(name="n")
        .query("n > 1")
    )

    print("duplicate (same flight, same days_left) rows:", len(dups))
    print(dups.head(10))
    # Plot curve
    plt.figure(figsize=(9, 5))
    plt.plot(curve["days_left"], curve["predicted_price"], marker="o")
    plt.axvline(best_day, color="red", linestyle="--", label="Best day")
    plt.xlabel("Days Left")
    plt.ylabel("Predicted Price")
    plt.title("Model-based Expected Price vs Days Left")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()
    
    eval_df = evaluate_policy(df_sorted, model, feature_cols, min_days=20, agg_price="mean")

    print(eval_df[["day_error","regret","saving_vs_now"]].describe())
    print("\n% flights saving money:", (eval_df["saving_vs_now"] > 0).mean() * 100)