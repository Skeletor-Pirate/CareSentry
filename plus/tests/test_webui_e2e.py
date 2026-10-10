"""End-to-end frontend verification test for CareSentry-AI Plus Web UI.
Connects to the live Gradio server via gradio_client.Client and verifies
all UI payloads, telemetry, JSON cards, and interactive heatmaps for both
Qwen and Gemma architectures.
"""

import json
import socket
import pytest

gradio_client = pytest.importorskip("gradio_client", reason="gradio_client is not installed")
Client = gradio_client.Client


def _is_server_online(host="127.0.0.1", port=7860) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _is_server_online(),
    reason="Gradio dev server is not running on 127.0.0.1:7860",
)


def test_gradio_frontend_audit_e2e():
    client = Client("http://127.0.0.1:7860")
    
    # 1. Test failing summary with safety violations and attention probe on Qwen
    raw_ehr = (
        "Patient is a 74-year-old female admitted for acute decompensation of chronic heart failure.\n"
        "Past Medical History: Chronic Kidney Disease stage 4, Hypertension, Atrial Fibrillation.\n"
        "LABS ON ADMISSION: Serum Creatinine 2.8 mg/dL, estimated GFR (eGFR) is 22 mL/min/1.73m2. Potassium 4.8 mEq/L.\n"
        "ACTIVE MEDICATIONS: Warfarin 4 mg daily for stroke prevention in AFib.\n"
        "DOCUMENTED ALLERGY: Patient has severe Penicillin allergy causing facial hives and dyspnea."
    )
    summary_content = (
        "Hospital Course & Discharge Summary:\n"
        "Discharge Orders:\n"
        "1. Continue home medications.\n"
        "2. Add Metformin 1000 mg BID for glycemic control.\n"
        "3. Prescribed Cephalexin 500 mg QID for 7 days to treat skin breakdown.\n"
        "4. Prescribed Naproxen 500 mg BID for lower back discomfort."
    )
    needle = "Patient has severe Penicillin allergy causing facial hives and dyspnea."
    
    result = client.predict(
        raw_ehr=raw_ehr,
        summary_content=summary_content,
        target_needle=needle,
        enable_attention_probe=True,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        api_name="/audit_discharge_summary",
    )
    
    verdict, base_json_str, plus_json_str, omissions_json_str, entities_json_str, heatmap_html = result
    
    # Assertions on UI outputs
    assert "[FAIL] AUDIT VERDICT: FAIL" in verdict
    assert "Qwen/Qwen2.5-1.5B-Instruct" in verdict
    assert "Cryptographic Block Hash:" in verdict
    
    base_data = json.loads(base_json_str)
    assert isinstance(base_data, dict)
    
    plus_violations = json.loads(plus_json_str)
    assert isinstance(plus_violations, list)
    assert len(plus_violations) > 0
    rules_flagged = [v.get("rule_id") for v in plus_violations]
    assert any("RENAL" in r or "METFORMIN" in r for r in rules_flagged)
    
    omissions = json.loads(omissions_json_str)
    assert isinstance(omissions, list)
    assert len(omissions) > 0
    
    entities = json.loads(entities_json_str)
    assert isinstance(entities, list)
    assert len(entities) > 0
    
    assert "heatmap-container" in heatmap_html or "caresentry" in heatmap_html or "<div" in heatmap_html
    assert len(heatmap_html) > 500


def test_gradio_frontend_audit_gemma():
    client = Client("http://127.0.0.1:7860")
    
    raw_ehr = "Patient with acute kidney injury. Documented allergy to Penicillin."
    summary = "Discharge on Amoxicillin 500mg TID."
    needle = "allergy to Penicillin"
    
    result = client.predict(
        raw_ehr=raw_ehr,
        summary_content=summary,
        target_needle=needle,
        enable_attention_probe=True,
        model_id="google/gemma-2-9b-it",
        api_name="/audit_discharge_summary",
    )
    
    verdict, base_json, plus_json, omissions_json, entities_json, heatmap_html = result
    assert "[FAIL]" in verdict
    assert "google/gemma-2-9b-it" in verdict
    assert len(heatmap_html) > 500


def test_gradio_frontend_audit_pass():
    client = Client("http://127.0.0.1:7860")
    
    raw_ehr = "Patient with mild hypertension. Blood pressure 130/85. No known allergies."
    clean_summary = "Discharge Summary: Patient recovered. Vital signs stable. Blood pressure 125/80. No acute distress."
    
    result = client.predict(
        raw_ehr=raw_ehr,
        summary_content=clean_summary,
        target_needle="",
        enable_attention_probe=False,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        api_name="/audit_discharge_summary",
    )
    
    verdict, base_json_str, plus_json_str, omissions_json_str, entities_json_str, heatmap_html = result
    
    assert "[PASS] AUDIT VERDICT: PASS" in verdict
    assert "All Invariants Satisfied" in verdict
    
    plus_violations = json.loads(plus_json_str)
    assert len(plus_violations) == 0
    
    omissions = json.loads(omissions_json_str)
    assert len(omissions) == 0


if __name__ == "__main__":
    print("Running test_gradio_frontend_audit_e2e (Qwen)...")
    test_gradio_frontend_audit_e2e()
    print("test_gradio_frontend_audit_e2e PASSED")
    
    print("Running test_gradio_frontend_audit_gemma (Gemma)...")
    test_gradio_frontend_audit_gemma()
    print("test_gradio_frontend_audit_gemma PASSED")

    print("Running test_gradio_frontend_audit_pass...")
    test_gradio_frontend_audit_pass()
    print("test_gradio_frontend_audit_pass PASSED")
    print("\nALL FRONTEND E2E TESTS (QWEN + GEMMA) PASSED SUCCESSFULLY!")
