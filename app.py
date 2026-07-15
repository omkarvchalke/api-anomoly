import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
import shap

app = FastAPI(title="Network Intrusion Detection API")

# --- Load artifacts once at startup ---
iso_model = joblib.load("model_artifacts/isolation_forest.joblib")

cat_model = CatBoostClassifier()
cat_model.load_model("model_artifacts/catboost_stage2.cbm")

with open("model_artifacts/proto_freq.json") as f:
    proto_freq_map = json.load(f)

with open("model_artifacts/feature_columns.json") as f:
    feature_meta = json.load(f)

ALL_FEATURES = feature_meta["all_features"]
NUMERIC_FEATURES = feature_meta["numeric_features"]
CAT_FEATURES = feature_meta["cat_features"]

# SHAP explainer, built once — reused across requests for speed
shap_explainer = shap.TreeExplainer(cat_model)


class FlowInput(BaseModel):
    """Raw flow features, matching the UNSW-NB15 schema after Phase 2 cleaning.
    srcip/dstip/attack_cat/Label/source_file/Stime/Ltime are intentionally excluded —
    they're either dropped during training or not available at inference time."""
    sport: float = Field(ge=0)
    dsport: float = Field(ge=0)
    proto: str
    state: str
    dur: float = Field(ge=0)
    sbytes: float = Field(ge=0)
    dbytes: float = Field(ge=0)
    sttl: float = Field(ge=0)
    dttl: float = Field(ge=0)
    sloss: float = Field(ge=0)
    dloss: float = Field(ge=0)
    service: str
    Sload: float = Field(ge=0)
    Dload: float = Field(ge=0)
    Spkts: float = Field(ge=0)
    Dpkts: float = Field(ge=0)
    swin: float = Field(ge=0)
    dwin: float = Field(ge=0)
    stcpb: float = Field(ge=0)
    dtcpb: float = Field(ge=0)
    smeansz: float = Field(ge=0)
    dmeansz: float = Field(ge=0)
    trans_depth: float = Field(ge=0)
    res_bdy_len: float = Field(ge=0)
    Sjit: float = Field(ge=0)
    Djit: float = Field(ge=0)
    Sintpkt: float = Field(ge=0)
    Dintpkt: float = Field(ge=0)
    tcprtt: float = Field(ge=0)
    synack: float = Field(ge=0)
    ackdat: float = Field(ge=0)
    is_sm_ips_ports: float = Field(ge=0, le=1)
    ct_state_ttl: float = Field(ge=0)
    ct_flw_http_mthd: float = Field(ge=0)
    is_ftp_login: float = Field(ge=0, le=1)
    ct_ftp_cmd: float = Field(ge=0)
    ct_srv_src: float = Field(ge=0)
    ct_srv_dst: float = Field(ge=0)
    ct_dst_ltm: float = Field(ge=0)
    ct_src_ltm: float = Field(ge=0, alias="ct_src_ ltm")  # matches the raw column's stray space
    ct_src_dport_ltm: float = Field(ge=0)
    ct_dst_sport_ltm: float = Field(ge=0)
    ct_dst_src_ltm: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)

    model_config = ConfigDict(populate_by_name=True)


def prepare_row(flow: FlowInput) -> pd.DataFrame:
    """Builds a single-row DataFrame matching training-time feature engineering."""
    row = flow.model_dump(by_alias=True)

    # Reproduce proto_freq — unseen protocols at inference get frequency 0
    row["proto_freq"] = proto_freq_map.get(row["proto"], 0.0)

    df_row = pd.DataFrame([row])
    return df_row[ALL_FEATURES]  # enforce exact column order the models expect


@app.post("/score")
def score_flow(flow: FlowInput):
    df_row = prepare_row(flow)

    # --- Stage 1: anomaly filter ---
    numeric_row = df_row[NUMERIC_FEATURES]
    is_anomaly = iso_model.predict(numeric_row)[0] == -1

    if not is_anomaly:
        return {
            "stage1_flagged": False,
            "prediction": "Normal",
            "risk_score": "low",
            "top_contributing_features": None
        }

    # --- Stage 2: attack categorization ---
    pool = Pool(df_row, cat_features=CAT_FEATURES)
    predicted_category = cat_model.predict(pool)[0][0]
    class_probs = cat_model.predict_proba(pool)[0]
    confidence = float(np.max(class_probs))

    # --- SHAP explanation for this specific prediction ---
    shap_vals = shap_explainer.shap_values(df_row)
    predicted_class_idx = list(cat_model.classes_).index(predicted_category)

    if isinstance(shap_vals, list):
        row_shap = shap_vals[predicted_class_idx][0]
    else:
        row_shap = shap_vals[0][:, predicted_class_idx]

    top_features_idx = np.argsort(np.abs(row_shap))[::-1][:5]
    top_features = [
        {"feature": ALL_FEATURES[i], "shap_value": float(row_shap[i])}
        for i in top_features_idx
    ]

    return {
        "stage1_flagged": True,
        "prediction": predicted_category,
        "confidence": round(confidence, 4),
        "risk_score": "high" if confidence > 0.7 else "medium",
        "top_contributing_features": top_features
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
