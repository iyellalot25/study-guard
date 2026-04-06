"""
Phase 5: API Test Suite

Tests every endpoint with valid, invalid, and edge-case inputs.
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

# Colour codes for terminal output
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {BLUE}→{RESET} {msg}")

def section(title):
    print(f"\n{YELLOW}{'=' * 60}{RESET}")
    print(f"{YELLOW}  {title}{RESET}")
    print(f"{YELLOW}{'=' * 60}{RESET}")

# ─────────────────────────────────────────────────────────────
# Sample inputs
# ─────────────────────────────────────────────────────────────

GOOD_STUDENT = {
    "total_study_hours":         18.0,
    "avg_session_length":        75.0,
    "study_sessions_per_week":   12.0,
    "focus_ratio":               0.85,
    "late_night_study_ratio":    0.05,
    "days_until_deadline_avg":   8.0,
    "video_completion_rate":     0.90,
    "practice_problem_attempts": 20.0,
    "forum_participation":       3.0,
    "study_streak_days":         14.0,
}

AT_RISK_STUDENT = {
    "total_study_hours":         6.0,
    "avg_session_length":        25.0,
    "study_sessions_per_week":   4.0,
    "focus_ratio":               0.40,
    "late_night_study_ratio":    0.65,
    "days_until_deadline_avg":   1.5,
    "video_completion_rate":     0.35,
    "practice_problem_attempts": 2.0,
    "forum_participation":       0.0,
    "study_streak_days":         2.0,
}

BORDERLINE_STUDENT = {
    "total_study_hours":         12.0,
    "avg_session_length":        45.0,
    "study_sessions_per_week":   8.0,
    "focus_ratio":               0.62,
    "late_night_study_ratio":    0.30,
    "days_until_deadline_avg":   3.0,
    "video_completion_rate":     0.60,
    "practice_problem_attempts": 8.0,
    "forum_participation":       1.0,
    "study_streak_days":         5.0,
}

passed = 0
failed = 0

def assert_ok(response, test_name, expected_status=200):
    global passed, failed
    if response.status_code == expected_status:
        data = response.json()
        if data.get('success'):
            ok(f"{test_name}  [{response.status_code}]  "
               f"{response.elapsed.microseconds // 1000}ms")
            passed += 1
            return data
        else:
            fail(f"{test_name}  [{response.status_code}]  "
                 f"success=false: {data.get('error')}")
            failed += 1
            return None
    else:
        fail(f"{test_name}  Expected {expected_status}, got {response.status_code}")
        failed += 1
        return None

def assert_error(response, test_name, expected_status=400):
    global passed, failed
    if response.status_code == expected_status:
        ok(f"{test_name}  [{response.status_code}] correctly rejected")
        passed += 1
        return response.json()
    else:
        fail(f"{test_name}  Expected {expected_status}, got {response.status_code}")
        failed += 1
        return None

# ─────────────────────────────────────────────────────────────
# TEST 1: Health check
# ─────────────────────────────────────────────────────────────
section("1 — GET /health")

r = requests.get(f"{BASE_URL}/health")
d = assert_ok(r, "Health check")
if d:
    info(f"Regressor:  {d.get('regressor')}")
    info(f"Classifier: {d.get('classifier')}")
    info(f"Threshold:  {d.get('threshold')}")

# ─────────────────────────────────────────────────────────────
# TEST 2: Feature info
# ─────────────────────────────────────────────────────────────
section("2 — GET /feature-info")

r = requests.get(f"{BASE_URL}/feature-info")
d = assert_ok(r, "Feature info")
if d:
    info(f"Features returned: {d.get('count')}")
    for name, meta in list(d['features'].items())[:3]:
        info(f"  {name}: range [{meta['min']}, {meta['max']}] {meta['unit']}")
    info(f"  ... and {d['count'] - 3} more")

# ─────────────────────────────────────────────────────────────
# TEST 3: Single predictions
# ─────────────────────────────────────────────────────────────
section("3 — POST /predict — Valid inputs")

for label, student in [
    ("Good student",       GOOD_STUDENT),
    ("At-risk student",    AT_RISK_STUDENT),
    ("Borderline student", BORDERLINE_STUDENT),
]:
    r = requests.post(
        f"{BASE_URL}/predict",
        json=student,
        headers={"Content-Type": "application/json"}
    )
    d = assert_ok(r, label)
    if d:
        pred = d['prediction']
        info(f"  Grade: {pred['predicted_grade']}  "
             f"Pass: {pred['pass_probability']}%  "
             f"Risk: {pred['risk_level']}")
        if pred['recommendations']:
            info(f"  Top rec: {pred['recommendations'][0]['message'][:60]}...")
        else:
            info(f"  No recommendations (student on track)")
        info(f"  Processing: {d.get('processing_time_ms')}ms")

# ─────────────────────────────────────────────────────────────
# TEST 4: Validation — bad inputs
# ─────────────────────────────────────────────────────────────
section("4 — POST /predict — Validation (should all reject)")

# Missing field
bad1 = {k: v for k, v in GOOD_STUDENT.items() if k != 'focus_ratio'}
r = requests.post(f"{BASE_URL}/predict", json=bad1)
d = assert_error(r, "Missing field (focus_ratio)")
if d: info(f"  Error: {d.get('error')}")

# Wrong type
bad2 = {**GOOD_STUDENT, "study_sessions_per_week": "lots"}
r = requests.post(f"{BASE_URL}/predict", json=bad2)
d = assert_error(r, "Wrong type (string instead of number)")
if d: info(f"  Error: {d.get('error')}")

# Out of range
bad3 = {**GOOD_STUDENT, "focus_ratio": 1.5}
r = requests.post(f"{BASE_URL}/predict", json=bad3)
d = assert_error(r, "Out of range (focus_ratio=1.5, max=1.0)")
if d: info(f"  Error: {d.get('error')}")

# Negative value
bad4 = {**GOOD_STUDENT, "total_study_hours": -5.0}
r = requests.post(f"{BASE_URL}/predict", json=bad4)
d = assert_error(r, "Negative value (study_hours=-5)")
if d: info(f"  Error: {d.get('error')}")

# Empty body
r = requests.post(f"{BASE_URL}/predict",
                  data="not json",
                  headers={"Content-Type": "application/json"})
d = assert_error(r, "Invalid JSON body")
if d: info(f"  Error: {d.get('error')}")

# ─────────────────────────────────────────────────────────────
# TEST 5: Batch prediction
# ─────────────────────────────────────────────────────────────
section("5 — POST /predict/batch")

# Valid batch
batch_valid = {"students": [GOOD_STUDENT, AT_RISK_STUDENT, BORDERLINE_STUDENT]}
r = requests.post(f"{BASE_URL}/predict/batch", json=batch_valid)
d = assert_ok(r, "Valid batch (3 students)")
if d:
    info(f"  Total: {d['total']}  Succeeded: {d['succeeded']}  Failed: {d['failed']}")
    for res in d['results']:
        if res:
            info(f"  Student {res['index']}: "
                 f"Grade={res['predicted_grade']}  "
                 f"Risk={res['risk_level']}")

# Mixed batch (1 valid, 1 invalid)
batch_mixed = {"students": [GOOD_STUDENT, {"focus_ratio": "bad"}]}
r = requests.post(f"{BASE_URL}/predict/batch", json=batch_mixed)
d = assert_ok(r, "Mixed batch (1 valid, 1 invalid)")
if d:
    info(f"  Succeeded: {d['succeeded']}  Failed: {d['failed']}")
    if d['errors']:
        info(f"  Error[{d['errors'][0]['index']}]: {d['errors'][0]['error']}")

# Empty batch
r = requests.post(f"{BASE_URL}/predict/batch", json={"students": []})
assert_error(r, "Empty batch")

# ─────────────────────────────────────────────────────────────
# TEST 6: Wrong HTTP methods
# ─────────────────────────────────────────────────────────────
section("6 — Wrong HTTP methods (should all reject)")

r = requests.post(f"{BASE_URL}/health")
assert_error(r, "POST /health (should be GET)", expected_status=405)

r = requests.get(f"{BASE_URL}/predict")
assert_error(r, "GET /predict (should be POST)", expected_status=405)

# ─────────────────────────────────────────────────────────────
# TEST 7: 404 — Non-existent endpoint
# ─────────────────────────────────────────────────────────────
section("7 — 404 — Non-existent endpoint")

r = requests.get(f"{BASE_URL}/nonexistent")
assert_error(r, "GET /nonexistent", expected_status=404)

# ─────────────────────────────────────────────────────────────
# TEST 8: SHAP contributions sanity check
# ─────────────────────────────────────────────────────────────
section("8 — SHAP contributions sanity check")

r = requests.post(f"{BASE_URL}/predict", json=AT_RISK_STUDENT)
d = assert_ok(r, "SHAP contributions present in response")
if d and 'shap_contributions' in d:
    shap = d['shap_contributions']
    info(f"  Features with SHAP values: {len(shap)}")

    # Top positive and negative drivers
    sorted_shap = sorted(shap.items(), key=lambda x: x[1])
    info(f"  Biggest negative: {sorted_shap[0][0]} = {sorted_shap[0][1]}")
    info(f"  Biggest positive: {sorted_shap[-1][0]} = {sorted_shap[-1][1]}")

    # Sanity: good student should have better SHAP than at-risk
    r2  = requests.post(f"{BASE_URL}/predict", json=GOOD_STUDENT)
    d2  = r2.json()
    good_grade = d2['prediction']['predicted_grade']
    risk_grade = d['prediction']['predicted_grade']

    if good_grade > risk_grade:
        ok(f"Good student grade ({good_grade}) > At-risk grade ({risk_grade}) ✓")
        passed += 1
    else:
        fail(f"Unexpected: good_grade={good_grade} <= risk_grade={risk_grade}")
        failed += 1

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
section("SUMMARY")

total = passed + failed
print(f"\n  Tests passed: {GREEN}{passed}/{total}{RESET}")
if failed > 0:
    print(f"  Tests failed: {RED}{failed}/{total}{RESET}")
else:
    print(f"  {GREEN}All tests passed! ✓{RESET}")

print(f"\n  API is ready for Phase 6 (Browser Extension)\n")