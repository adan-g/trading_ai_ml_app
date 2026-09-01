import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import requests
from fastapi import FastAPI, Request
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


def clean_signal_for_supabase(signal: Dict[str, Any]):
    cleaned = signal.copy()

    # Do not save secret in Supabase
    cleaned.pop("secret", None)

     # Add received time like Google Sheets does
    if not cleaned.get("received_time"):
        cleaned["received_time"] = datetime.utcnow().isoformat()

    # Convert empty strings to None
    for key, value in list(cleaned.items()):
        if value == "":
            cleaned[key] = None

    return cleaned


def get_completed_signals():
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"

    params = {
        "select": "*",
        "order": "date_time.desc",
    }

    response = requests.get(
        url,
        headers=supabase_headers(),
        params=params,
    )

    if response.status_code not in [200, 201]:
        raise Exception(f"Supabase error: {response.status_code} - {response.text}")

    rows = response.json()

    completed_rows = []

    for row in rows:
        result = str(row.get("result", "")).strip().upper()

        if result in ["WIN", "LOSS"]:
            row["result"] = result
            completed_rows.append(row)

    return completed_rows


def insert_raw_signal(signal: Dict[str, Any]):
    clean_signal = clean_signal_for_supabase(signal)

    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"

    response = requests.post(
        url,
        headers={
            **supabase_headers(),
            "Prefer": "return=minimal",
        },
        data=json.dumps(clean_signal),
    )

    return response.status_code, response.text


def update_signal_result(id_trade: str, signal: Dict[str, Any]):
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"

    params = {
        "id_trade": f"eq.{id_trade}"
    }

    payload = {
        "signal_type": "ENTRY + RESULT",
        "result": signal.get("result"),
        "r_result": signal.get("r_result"),
        "candles_to_result": signal.get("candles_to_result"),
        "tracking_target_2r": signal.get("tracking_target_2r"),
        "target_price_2r": signal.get("target_price_2r"),
        "reached_2r": signal.get("reached_2r"),
        "candles_to_2r": signal.get("candles_to_2r"),
        "max_r_before_sl": signal.get("max_r_before_sl"),
        "updated_at": datetime.utcnow().isoformat(),
    }

    # Convert empty strings to None
    payload = {
        key: None if value == "" else value
        for key, value in payload.items()
    }

    # Normalize result values
    if payload.get("result") is not None:
        payload["result"] = str(payload["result"]).strip().upper()

    if payload.get("reached_2r") is not None:
        payload["reached_2r"] = str(payload["reached_2r"]).strip().upper()

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
            lambda x: x if x in known_classes else encoder.classes_[0]
        )

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
        return {
            "status": "error",
            "message": "DISCORD_WEBHOOK_URL missing"
        }

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

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "TradingMLFilter/1.0"
    }

    response = requests.post(
        DISCORD_WEBHOOK_URL,
        headers=headers,
        json=payload,
        timeout=10
    )

    return {
        "discord_status_code": response.status_code,
        "discord_response": response.text
    }


# ====================================================
# ROUTES
# ====================================================

@app.get("/test-discord")
def test_discord():
    if not DISCORD_WEBHOOK_URL:
        return {
            "status": "error",
            "message": "DISCORD_WEBHOOK_URL is missing in Render environment variables."
        }

    payload = {
        "content": "✅ Discord test from Render ML app."
    }

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "TradingMLFilter/1.0"
    }

    response = requests.post(
        DISCORD_WEBHOOK_URL,
        headers=headers,
        json=payload,
        timeout=10
    )

    return {
        "status": "sent" if response.status_code in [200, 204] else "error",
        "discord_status_code": response.status_code,
        "discord_response": response.text
    }

@app.get("/")
def home():
    return {
        "status": "online",
        "app": "Trading ML Filter",
        "model": "Random Forest",
        "routes": {
            "train": "/train",
            "webhook": "/webhook",
            "health": "/health",
            "test_supabase": "/test-supabase",
            "debug_counts": "/debug-counts",
        }
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

    df["result"] = df["result"].astype(str).str.strip().str.upper()
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

    if len(y.unique()) < 2:
        return {
            "status": "not_enough_classes",
            "message": "Need both WIN and LOSS trades to train.",
            "rows_found": len(df),
        }

    if len(df) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
    else:
        X_train = X
        X_test = X
        y_train = y
        y_test = y

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
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
        "win_count": int((df["target"] == 1).sum()),
        "loss_count": int((df["target"] == 0).sum()),
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

    signal_type = str(signal.get("signal_type", "")).upper().strip()
    id_trade = signal.get("id_trade")

    if not id_trade:
        return {
            "status": "error",
            "message": "Missing id_trade"
        }

    # ====================================================
    # RESULT ALERT
    # Update existing Supabase row with WIN/LOSS result
    # ====================================================
    if signal_type == "RESULT":
        update_status, update_response = update_signal_result(id_trade, signal)

        return {
            "status": "result_processed",
            "id_trade": id_trade,
            "symbol": signal.get("symbol"),
            "side": signal.get("side"),
            "result": signal.get("result"),
            "supabase_update_status": update_status,
            "supabase_update_response": update_response,
        }

    # ====================================================
    # ENTRY ALERT
    # Score entry with Random Forest, insert row, send Discord if passed
    # ====================================================
    if signal_type != "ENTRY":
        return {
            "status": "ignored",
            "message": f"Signal type ignored: {signal_type}"
        }

    model, encoders = load_model()

    if model is None or encoders is None:
        return {
            "status": "no_model",
            "message": "Train the model first using /train."
        }

    X_signal = prepare_single_signal(signal, encoders)

    probability = float(model.predict_proba(X_signal)[0][1])

    decision = "SEND" if probability >= ML_THRESHOLD else "SKIP"

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
        "status": "entry_processed",
        "id_trade": id_trade,
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


@app.get("/test-supabase")
def test_supabase():
    try:
        url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
        params = {
            "select": "id,id_trade,symbol,result,ml_probability,ml_decision",
            "limit": "5",
            "order": "created_at.desc",
        }

        response = requests.get(
            url,
            headers=supabase_headers(),
            params=params,
        )

        return {
            "status": "connected" if response.status_code == 200 else "error",
            "supabase_status_code": response.status_code,
            "supabase_response": response.json() if response.text else [],
            "raw_response": response.text,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/debug-counts")
def debug_counts():
    try:
        url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"

        response = requests.get(
            url,
            headers=supabase_headers(),
            params={
                "select": "id_trade,symbol,signal_type,result,reached_2r,ml_probability,ml_decision,created_at",
                "limit": "20",
                "order": "created_at.desc",
            },
        )

        if response.status_code != 200:
            return {
                "status": "error",
                "code": response.status_code,
                "response": response.text,
            }

        recent_rows = response.json()

        all_response = requests.get(
            url,
            headers=supabase_headers(),
            params={
                "select": "result,signal_type,ml_decision",
            },
        )

        result_counts = {}
        signal_type_counts = {}
        ml_decision_counts = {}

        if all_response.status_code == 200:
            rows = all_response.json()

            for row in rows:
                result = str(row.get("result", "NULL")).strip()
                signal_type = str(row.get("signal_type", "NULL")).strip()
                ml_decision = str(row.get("ml_decision", "NULL")).strip()

                if result == "":
                    result = "EMPTY"

                if signal_type == "":
                    signal_type = "EMPTY"

                if ml_decision == "":
                    ml_decision = "EMPTY"

                result_counts[result] = result_counts.get(result, 0) + 1
                signal_type_counts[signal_type] = signal_type_counts.get(signal_type, 0) + 1
                ml_decision_counts[ml_decision] = ml_decision_counts.get(ml_decision, 0) + 1

        return {
            "status": "connected",
            "recent_rows": recent_rows,
            "result_counts": result_counts,
            "signal_type_counts": signal_type_counts,
            "ml_decision_counts": ml_decision_counts,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }