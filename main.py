import os
import json
import pickle
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# ====================================================
# CONFIG
# ====================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "my_secret_123")

TABLE_NAME = "signals"
MODEL_FILE = "random_forest_model.pkl"
ENCODERS_FILE = "label_encoders.pkl"

ML_THRESHOLD = float(os.getenv("ML_THRESHOLD", "0.60"))

app = FastAPI(title="Trading ML Filter")


# ====================================================
# FEATURES USED BY RANDOM FOREST
# ====================================================

FEATURE_COLUMNS = [
    "side",
    "timeframe",
    "risk_reward",
    "fast_ema",
    "slow_ema",
    "ema_distance",
    "ema_distance_atr",
    "atr",
    "rsi",
    "rsi_zone",
    "rsi_above_50",
    "rsi_slope",
    "candle_body",
    "upper_wick",
    "lower_wick",
    "touching_bsl",
    "touching_ssl",
    "closest_bsl_distance_atr",
    "closest_ssl_distance_atr",
    "closest_zone_type",
    "closest_zone_age",
    "active_bsl_count",
    "active_ssl_count",
    "session",
    "day_of_week",
    "hour",
    "trend_1h",
    "trend_4h",
]


CATEGORICAL_COLUMNS = [
    "side",
    "timeframe",
    "rsi_zone",
    "rsi_above_50",
    "touching_bsl",
    "touching_ssl",
    "closest_zone_type",
    "session",
    "day_of_week",
    "trend_1h",
    "trend_4h",
]


NUMERIC_COLUMNS = [
    "risk_reward",
    "fast_ema",
    "slow_ema",
    "ema_distance",
    "ema_distance_atr",
    "atr",
    "rsi",
    "rsi_slope",
    "candle_body",
    "upper_wick",
    "lower_wick",
    "closest_bsl_distance_atr",
    "closest_ssl_distance_atr",
    "closest_zone_age",
    "active_bsl_count",
    "active_ssl_count",
    "hour",
]


# ====================================================
# SUPABASE HELPERS
# ====================================================

def supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def get_completed_signals():
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    params = {
        "select": "*",
        "result": "in.(WIN,LOSS)",
        "order": "date_time.desc",
    }

    response = requests.get(url, headers=supabase_headers(), params=params)

    if response.status_code not in [200, 201]:
        raise Exception(f"Supabase error: {response.status_code} - {response.text}")

    return response.json()


def update_signal_ml_result(id_trade: str, probability: float, decision: str):
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    params = {
        "id_trade": f"eq.{id_trade}"
    }

    payload = {
        "ml_probability": probability,
        "ml_decision": decision,
        "model_version": "random_forest_v1",
        "updated_at": datetime.utcnow().isoformat(),
    }

    response = requests.patch(
        url,
        headers={
            **supabase_headers(),
            "Prefer": "return=minimal",
        },
        params=params,
        data=json.dumps(payload),
    )

    return response.status_code, response.text


def insert_raw_signal(signal: Dict[str, Any]):
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"

    response = requests.post(
        url,
        headers={
            **supabase_headers(),
            "Prefer": "return=minimal",
        },
        data=json.dumps(signal),
    )

    return response.status_code, response.text


# ====================================================
# DATA CLEANING
# ====================================================

def clean_dataframe(df: pd.DataFrame):
    df = df.copy()

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0)

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str).fillna("UNKNOWN")
        df[col] = df[col].replace(["", "None", "nan", "NaN"], "UNKNOWN")

    return df


def train_encoders(df: pd.DataFrame):
    encoders = {}

    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder

    return df, encoders


def apply_encoders_to_signal(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]):
    df = df.copy()

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str).fillna("UNKNOWN")
        df[col] = df[col].replace(["", "None", "nan", "NaN"], "UNKNOWN")

        encoder = encoders[col]

        known_classes = set(encoder.classes_)

        df[col] = df[col].apply(
            lambda x: x if x in known_classes else "UNKNOWN"
        )

        if "UNKNOWN" not in known_classes:
            encoder.classes_ = list(encoder.classes_) + ["UNKNOWN"]

        df[col] = encoder.transform(df[col].astype(str))

    return df


def prepare_single_signal(signal: Dict[str, Any], encoders: Dict[str, LabelEncoder]):
    df = pd.DataFrame([signal])

    df = clean_dataframe(df)
    df = apply_encoders_to_signal(df, encoders)

    return df[FEATURE_COLUMNS]


# ====================================================
# MODEL HELPERS
# ====================================================

def save_model(model, encoders):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    with open(ENCODERS_FILE, "wb") as f:
        pickle.dump(encoders, f)


def load_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODERS_FILE):
        return None, None

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    with open(ENCODERS_FILE, "rb") as f:
        encoders = pickle.load(f)

    return model, encoders


# ====================================================
# DISCORD
# ====================================================

def send_discord_alert(signal: Dict[str, Any], probability: float, decision: str):
    if not DISCORD_WEBHOOK_URL:
        return

    symbol = signal.get("symbol", "")
    side = signal.get("side", "")
    timeframe = signal.get("timeframe", "")
    entry = signal.get("entry_price", "")
    stop = signal.get("stop_loss", "")
    tp = signal.get("take_profit", "")
    trend_1h = signal.get("trend_1h", "")
    trend_4h = signal.get("trend_4h", "")

    message = f"""
🚨 **ML FILTER PASSED**

**Symbol:** {symbol}
**Side:** {side}
**Timeframe:** {timeframe}

**Entry:** {entry}
**Stop Loss:** {stop}
**Take Profit:** {tp}

**Trend 1H:** {trend_1h}
**Trend 4H:** {trend_4h}

**Random Forest Probability:** {round(probability * 100, 2)}%
**Decision:** {decision}
"""

    payload = {
        "content": message
    }

    requests.post(DISCORD_WEBHOOK_URL, json=payload)


# ====================================================
# ROUTES
# ====================================================

@app.get("/")
def home():
    return {
        "status": "online",
        "app": "Trading ML Filter",
        "model": "Random Forest",
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.api_route("/train", methods=["GET", "POST"])
def train_model():
    rows = get_completed_signals()

    if len(rows) < 5:
        return {
            "status": "not_enough_data",
            "message": "Need at least 5 completed trades to train.",
            "rows_found": len(rows),
        }

    df = pd.DataFrame(rows)

    df = df[df["result"].isin(["WIN", "LOSS"])]

    if len(df) < 5:
        return {
            "status": "not_enough_data",
            "message": "Need at least 5 WIN/LOSS trades.",
            "rows_found": len(df),
        }

    df = clean_dataframe(df)

    df["target"] = df["result"].apply(lambda x: 1 if x == "WIN" else 0)

    df, encoders = train_encoders(df)

    X = df[FEATURE_COLUMNS]
    y = df["target"]

    if len(df) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y if len(y.unique()) > 1 else None,
        )
    else:
        X_train = X
        X_test = X
        y_train = y
        y_test = y

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    save_model(model, encoders)

    feature_importance = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "status": "trained",
        "rows_used": len(df),
        "accuracy": round(float(accuracy), 4),
        "top_features": [
            {
                "feature": feature,
                "importance": round(float(importance), 4)
            }
            for feature, importance in feature_importance[:10]
        ],
    }


@app.post("/webhook")
async def tradingview_webhook(request: Request):
    signal = await request.json()

    if signal.get("secret") != WEBHOOK_SECRET:
        return {
            "status": "unauthorized"
        }

    signal_type = str(signal.get("signal_type", "")).upper()

    if signal_type != "ENTRY":
        return {
            "status": "ignored",
            "message": "Only ENTRY signals are filtered by ML."
        }

    model, encoders = load_model()

    if model is None or encoders is None:
        return {
            "status": "no_model",
            "message": "Train the model first using /train."
        }

    X_signal = prepare_single_signal(signal, encoders)

    probability = float(model.predict_proba(X_signal)[0][1])

    if probability >= ML_THRESHOLD:
        decision = "SEND"
    else:
        decision = "SKIP"

    signal["ml_probability"] = probability
    signal["ml_decision"] = decision
    signal["model_version"] = "random_forest_v1"

    try:
        insert_status, insert_response = insert_raw_signal(signal)
    except Exception as e:
        insert_status = "error"
        insert_response = str(e)

    if decision == "SEND":
        send_discord_alert(signal, probability, decision)

    return {
        "status": "processed",
        "id_trade": signal.get("id_trade"),
        "symbol": signal.get("symbol"),
        "side": signal.get("side"),
        "probability": round(probability, 4),
        "decision": decision,
        "supabase_insert_status": insert_status,
        "supabase_insert_response": insert_response,
    }


@app.post("/score-test")
async def score_test(request: Request):
    signal = await request.json()

    model, encoders = load_model()

    if model is None or encoders is None:
        return {
            "status": "no_model",
            "message": "Train the model first using /train."
        }

    X_signal = prepare_single_signal(signal, encoders)

    probability = float(model.predict_proba(X_signal)[0][1])

    decision = "SEND" if probability >= ML_THRESHOLD else "SKIP"

    return {
        "status": "scored",
        "probability": round(probability, 4),
        "decision": decision,
    }