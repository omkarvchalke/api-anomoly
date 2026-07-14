import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

# A baseline "normal-looking" template — each test case overrides only the
# fields relevant to that scenario, so differences are easy to trace
BASE = {
    "sport": 53, "dsport": 53, "proto": "udp", "state": "CON",
    "dur": 0.05, "sbytes": 132, "dbytes": 164, "sttl": 254, "dttl": 252,
    "sloss": 0, "dloss": 0, "service": "dns", "Sload": 100000, "Dload": 120000,
    "Spkts": 2, "Dpkts": 2, "swin": 0, "dwin": 0, "stcpb": 0, "dtcpb": 0,
    "smeansz": 66, "dmeansz": 82, "trans_depth": 0, "res_bdy_len": 0,
    "Sjit": 0, "Djit": 0, "Sintpkt": 0.5, "Dintpkt": 0.5, "tcprtt": 0,
    "synack": 0, "ackdat": 0, "is_sm_ips_ports": 0, "ct_state_ttl": 1,
    "ct_flw_http_mthd": 0, "is_ftp_login": 0, "ct_ftp_cmd": 0,
    "ct_srv_src": 1, "ct_srv_dst": 1, "ct_dst_ltm": 1, "ct_src_ ltm": 1,
    "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 1, "hour": 14
}


def case(overrides):
    """Returns a full payload = BASE with specific fields overridden."""
    payload = BASE.copy()
    payload.update(overrides)
    return payload


# Cases confirmed against the current model artifacts to NOT trip the
# isolation-forest anomaly filter. "block cipher pattern" reads like an
# attack but the model doesn't flag it at these feature values — that's
# current ground truth, not a guess, so it belongs here rather than below.
NORMAL_CASES = [
    ("normal_dns_query", case({})),
    ("normal_http_browsing", case({
        "proto": "tcp", "service": "http", "state": "FIN",
        "dur": 1.2, "sbytes": 800, "dbytes": 4200, "sttl": 62, "dttl": 252,
        "Spkts": 6, "Dpkts": 8
    })),
    ("generic_block_cipher_pattern", case({
        "proto": "tcp", "service": "-", "state": "FIN", "sttl": 31,
        "ct_state_ttl": 2, "sbytes": 528, "dbytes": 304
    })),
    ("all_zero_payload", case({
        k: 0 for k in BASE if isinstance(BASE[k], (int, float))
    })),
]

# Cases confirmed to trip the isolation-forest filter and reach stage 2.
# These are hand-crafted feature combos, not real UNSW-NB15 samples, so the
# predicted *category* isn't asserted — only that stage 2 runs and returns
# a well-formed classification + explanation.
ANOMALOUS_CASES = [
    ("dos_syn_flood_pattern", case({
        "proto": "tcp", "state": "INT", "sttl": 62, "ct_state_ttl": 3,
        "dur": 0.001, "sbytes": 60, "dbytes": 0, "Spkts": 1, "Dpkts": 0,
        "ct_srv_src": 45, "ct_dst_ltm": 40
    })),
    ("exploits_buffer_overflow_attempt", case({
        "proto": "tcp", "service": "http", "state": "CON",
        "smeansz": 1500, "dmeansz": 40, "trans_depth": 3, "res_bdy_len": 8000,
        "sbytes": 9000, "dur": 2.5
    })),
    ("fuzzers_malformed_packet", case({
        "proto": "tcp", "service": "-", "state": "CON",
        "Sjit": 950.2, "Djit": 430.7, "dsport": 49213, "sbytes": 3, "dbytes": 3,
        "smeansz": 3, "dmeansz": 3
    })),
    ("reconnaissance_port_scan", case({
        "proto": "tcp", "state": "REQ", "dur": 0.0001, "sbytes": 0, "dbytes": 0,
        "Spkts": 1, "Dpkts": 0, "ct_dst_sport_ltm": 35, "ct_src_dport_ltm": 32,
        "ct_srv_dst": 30
    })),
    ("shellcode_injection_attempt", case({
        "proto": "tcp", "service": "http", "state": "CON",
        "res_bdy_len": 15000, "trans_depth": 5, "sbytes": 16000, "smeansz": 1400
    })),
    ("worms_self_propagating", case({
        "proto": "tcp", "state": "CON", "is_sm_ips_ports": 1,
        "ct_srv_src": 20, "ct_srv_dst": 20, "ct_dst_src_ltm": 18, "dur": 0.2
    })),
    ("backdoor_suspicious_ftp_login", case({
        "sport": 21, "proto": "tcp", "state": "FIN", "sttl": 62,
        "is_ftp_login": 1, "ct_ftp_cmd": 4, "ct_state_ttl": 3
    })),
    ("analysis_http_method_probing", case({
        "proto": "tcp", "service": "http", "state": "CON",
        "ct_flw_http_mthd": 6, "dur": 0.8
    })),
    ("edge_extreme_large_values", case({
        "sbytes": 999999999, "dbytes": 999999999, "Spkts": 500000, "Dpkts": 500000,
        "Sload": 1e12, "Dload": 1e12, "dur": 99999.9
    })),
    ("edge_unseen_protocol_and_service", case({
        "proto": "sctp", "service": "unknown_svc", "state": "XYZ"
    })),
]


class TestHealthCheck:
    def test_health_check_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestNormalTraffic:
    @pytest.mark.parametrize("name,payload", NORMAL_CASES, ids=[c[0] for c in NORMAL_CASES])
    def test_not_flagged_and_short_circuits(self, name, payload):
        resp = client.post("/score", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["stage1_flagged"] is False
        assert body["prediction"] == "Normal"
        assert body["risk_score"] == "low"
        assert body["top_contributing_features"] is None
        assert "confidence" not in body


class TestAnomalousTraffic:
    @pytest.mark.parametrize("name,payload", ANOMALOUS_CASES, ids=[c[0] for c in ANOMALOUS_CASES])
    def test_flagged_with_well_formed_explanation(self, name, payload):
        resp = client.post("/score", json=payload)
        assert resp.status_code == 200
        body = resp.json()

        assert body["stage1_flagged"] is True
        assert isinstance(body["prediction"], str) and body["prediction"]
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["risk_score"] == ("high" if body["confidence"] > 0.7 else "medium")

        top = body["top_contributing_features"]
        assert isinstance(top, list) and len(top) == 5
        magnitudes = [abs(f["shap_value"]) for f in top]
        assert magnitudes == sorted(magnitudes, reverse=True)
        for entry in top:
            assert set(entry.keys()) == {"feature", "shap_value"}
            assert isinstance(entry["feature"], str)
            assert isinstance(entry["shap_value"], float)


class TestInputValidation:
    def test_negative_duration_rejected(self):
        resp = client.post("/score", json=case({"dur": -5.0}))
        assert resp.status_code == 422
        fields = {e["loc"][-1] for e in resp.json()["detail"]}
        assert "dur" in fields

    def test_negative_bytes_rejected(self):
        resp = client.post("/score", json=case({"sbytes": -100}))
        assert resp.status_code == 422
        fields = {e["loc"][-1] for e in resp.json()["detail"]}
        assert "sbytes" in fields

    def test_hour_above_range_rejected(self):
        resp = client.post("/score", json=case({"hour": 27}))
        assert resp.status_code == 422

    def test_hour_negative_rejected(self):
        resp = client.post("/score", json=case({"hour": -1}))
        assert resp.status_code == 422

    def test_binary_flag_out_of_range_rejected(self):
        resp = client.post("/score", json=case({"is_sm_ips_ports": 2}))
        assert resp.status_code == 422

    def test_missing_required_field_rejected(self):
        payload = case({})
        del payload["proto"]
        resp = client.post("/score", json=payload)
        assert resp.status_code == 422

    def test_missing_aliased_field_rejected(self):
        payload = case({})
        del payload["ct_src_ ltm"]
        resp = client.post("/score", json=payload)
        assert resp.status_code == 422
