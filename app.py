"""
Phase 5: Flask API — Model Serving Backend

Endpoints:
  GET  /health           → Server status check
  GET  /feature-info     → Feature names, ranges, descriptions
  POST /predict          → Single student prediction + recommendations
  POST /predict/batch    → Multiple students at once
"""

import numpy as np
import pandas as pd
import pickle
import os
import time
import logging
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS


# APP SETUP

app = Flask(__name__)

# CORS: allows the browser extension (different origin) to call this API
# Without this, the browser blocks cross-origin requests by default
CORS(app)

# Logging: record every request and error to a file + console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# LOAD MODEL ARTIFACTS (once at startup, not per request)
# Loading models is expensive (~100ms). We do it once when the server starts,
# then reuse the same objects for every incoming request.

logger.info("Loading model artifacts...")

MODEL_PATH       = 'trained_models.pkl'
EXPLAIN_PATH     = 'explainability.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"'{MODEL_PATH}' not found. Run phases 1-4 first."
    )
if not os.path.exists(EXPLAIN_PATH):
    raise FileNotFoundError(
        f"'{EXPLAIN_PATH}' not found. Run phase4_explainability.py first."
    )

# trained_models.pkl — pure ML artifacts (models + scaler)
with open(MODEL_PATH, 'rb') as f:
    artifacts = pickle.load(f)

REGRESSOR        = artifacts['regressor']
CLASSIFIER       = artifacts['classifier']
SCALER           = artifacts['scaler']
THRESHOLD        = artifacts['best_threshold']
FEATURE_COLS     = artifacts['feature_columns']

# explainability.pkl — SHAP explainers (loaded separately, used for per-request SHAP)
with open(EXPLAIN_PATH, 'rb') as f:
    explain_artifacts = pickle.load(f)

# Use the real SHAP TreeExplainer if it was saved
# Fall back to feature importances as a proxy if not available
REG_EXPLAINER    = explain_artifacts.get('reg_explainer', None)
SHAP_AVAILABLE   = explain_artifacts.get('shap_available', False)
REG_MEAN_SHAP    = explain_artifacts.get('reg_mean_shap', None)  # global importances

# Feature names after scaling + interactions (12 total)
# The scaler only covers the original 10 — interactions are added after
ORIGINAL_FEATURES = [f for f in FEATURE_COLS
                     if f not in ('study_efficiency', 'procrastination_penalty')]

logger.info(f"Loaded models:  {artifacts['regressor_name']} + {artifacts['classifier_name']}")
logger.info(f"Threshold:      {THRESHOLD:.3f}")
logger.info(f"SHAP explainer: {'TreeExplainer (exact)' if SHAP_AVAILABLE else 'feature importance proxy'}")
logger.info(f"Features expected: {ORIGINAL_FEATURES}")

# FEATURE METADATA
# Used for input validation AND by the /feature-info endpoint
# The browser extension reads this to know what to send

FEATURE_META = {
    'total_study_hours': {
        'description': 'Total weekly study hours on educational sites',
        'min': 0, 'max': 80, 'unit': 'hours/week',
        'example': 15.0
    },
    'avg_session_length': {
        'description': 'Average length of a single study session',
        'min': 5, 'max': 300, 'unit': 'minutes',
        'example': 60.0
    },
    'study_sessions_per_week': {
        'description': 'Number of distinct study sessions per week',
        'min': 0, 'max': 50, 'unit': 'sessions/week',
        'example': 10.0
    },
    'focus_ratio': {
        'description': 'Fraction of study time without context switches',
        'min': 0.0, 'max': 1.0, 'unit': 'ratio',
        'example': 0.75
    },
    'late_night_study_ratio': {
        'description': 'Fraction of study time after 10 PM',
        'min': 0.0, 'max': 1.0, 'unit': 'ratio',
        'example': 0.1
    },
    'days_until_deadline_avg': {
        'description': 'Average days before deadline when work starts',
        'min': 0, 'max': 30, 'unit': 'days',
        'example': 5.0
    },
    'video_completion_rate': {
        'description': 'Fraction of educational videos watched fully',
        'min': 0.0, 'max': 1.0, 'unit': 'ratio',
        'example': 0.8
    },
    'practice_problem_attempts': {
        'description': 'Number of practice problems attempted per week',
        'min': 0, 'max': 200, 'unit': 'attempts/week',
        'example': 10.0
    },
    'forum_participation': {
        'description': 'Number of forum posts or answers per week',
        'min': 0, 'max': 50, 'unit': 'posts/week',
        'example': 2.0
    },
    'study_streak_days': {
        'description': 'Longest consecutive study streak in days',
        'min': 0, 'max': 365, 'unit': 'days',
        'example': 7.0
    },
}


# RECOMMENDATION ENGINE (defined here, not loaded from pkl)

# Keeping business logic in the API — not pickled

def _cond_low(v):  return v < -0.3
def _cond_high(v): return v >  0.3
def _cond_vlow(v): return v < -0.5

RECOMMENDATION_RULES = [
    {
        'feature':   'total_study_hours',
        'condition': _cond_low,
        'message':   (
            "📚 Study more consistently. You're below average on weekly study hours. "
            "Aim for at least 15 hours/week — even 2 extra hours daily adds up."
        ),
    },
    {
        'feature':   'focus_ratio',
        'condition': _cond_low,
        'message':   (
            "🎯 Reduce distractions during study sessions. "
            "Try the Pomodoro technique: 25 min focused work, 5 min break. "
            "Your current focus ratio is below average."
        ),
    },
    {
        'feature':   'late_night_study_ratio',
        'condition': _cond_high,
        'message':   (
            "😴 Shift study time earlier in the day. Late-night cramming reduces "
            "retention significantly. Studies show morning/afternoon studying "
            "improves recall by up to 20%."
        ),
    },
    {
        'feature':   'avg_session_length',
        'condition': _cond_low,
        'message':   (
            "⏱️ Extend your study sessions. Short sessions (under 30 min) don't "
            "allow deep focus. Aim for 45-90 minute sessions with proper breaks."
        ),
    },
    {
        'feature':   'practice_problem_attempts',
        'condition': _cond_low,
        'message':   (
            "✏️ Do more practice problems. Active recall (solving problems) is "
            "far more effective than passive reading. Aim for at least 10 "
            "practice problems per week."
        ),
    },
    {
        'feature':   'video_completion_rate',
        'condition': _cond_low,
        'message':   (
            "🎥 Complete your educational videos. You're starting videos but not "
            "finishing them. Incomplete content leads to knowledge gaps — "
            "try shorter videos or set completion goals."
        ),
    },
    {
        'feature':   'days_until_deadline_avg',
        'condition': _cond_low,
        'message':   (
            "📅 Start assignments earlier. You're working too close to deadlines. "
            "Beginning work 5+ days before submission improves quality and "
            "reduces stress significantly."
        ),
    },
    {
        'feature':   'study_streak_days',
        'condition': _cond_low,
        'message':   (
            "🔥 Build a study streak. Consistency beats intensity. "
            "Even 30 minutes of daily study is better than 4-hour weekend sessions. "
            "Try not to miss more than one day in a row."
        ),
    },
    {
        'feature':   'forum_participation',
        'condition': _cond_vlow,
        'message':   (
            "💬 Engage with the learning community. Asking questions and "
            "discussing concepts with peers solidifies understanding. "
            "Try posting at least one question or answer per week."
        ),
    },
    {
        'feature':   'study_efficiency',
        'condition': _cond_low,
        'message':   (
            "⚡ Improve study efficiency (hours × focus). You might be studying "
            "enough hours but getting distracted, OR staying focused but not long "
            "enough. Focus on quality time: fewer distractions, longer sessions."
        ),
    },
    {
        'feature':   'procrastination_penalty',
        'condition': _cond_high,
        'message':   (
            "🚨 You're showing a procrastination pattern: late-night work near "
            "deadlines. Plan your week on Sundays — assign specific subjects to "
            "specific days so you're never scrambling at the last minute."
        ),
    },
]


def generate_recommendations(student_features, shap_values_reg,
                              predicted_grade, pass_probability,
                              threshold=0.44, top_n=3):
    """Generate SHAP-ranked personalised recommendations for one student."""
    shap_series = pd.Series(shap_values_reg, index=FEATURE_COLS)
    triggered   = []

    for rule in RECOMMENDATION_RULES:
        feat   = rule['feature']
        val    = student_features[feat]
        shap_v = float(shap_series[feat])
        if rule['condition'](val):
            triggered.append({
                'feature':  feat,
                'message':  rule['message'],
                'shap':     shap_v,
                'priority': abs(shap_v),
            })

    triggered.sort(key=lambda x: x['priority'], reverse=True)

    # Risk calculation: prioritize predicted grade, then probability
    # This makes risk labels more intuitive for students
    if predicted_grade < 60:
        risk = 'HIGH RISK'           # Actually failing (grade < 60)
    elif pass_probability < threshold:
        risk = 'MODERATE RISK'       # Passing but model is uncertain
    elif pass_probability < threshold + 0.15:
        risk = 'MODERATE RISK'       # Passing but could improve
    else:
        risk = 'ON TRACK'            # Confidently passing


    return {
        'predicted_grade':  round(float(predicted_grade), 1),
        'pass_probability': round(pass_probability * 100, 1),
        'risk_level':       risk,
        'recommendations':  triggered[:top_n],
        'total_flags':      len(triggered),
    }

def validate_student_input(data: dict) -> tuple[dict | None, str | None]:
    """
    Validates and cleans raw input from the browser extension.

    Args:
        data: raw JSON dict from request

    Returns:
        (cleaned_dict, None)      on success
        (None, error_message)     on failure
    """
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object"

    cleaned = {}

    for feature, meta in FEATURE_META.items():
        # Check presence
        if feature not in data:
            return None, f"Missing required field: '{feature}'"

        val = data[feature]

        # Check type (must be numeric)
        if not isinstance(val, (int, float)):
            return None, f"Field '{feature}' must be a number, got {type(val).__name__}"

        # Check for NaN / Inf (can happen with JS float edge cases)
        if not np.isfinite(val):
            return None, f"Field '{feature}' must be a finite number"

        # Range check
        if not (meta['min'] <= val <= meta['max']):
            return None, (
                f"Field '{feature}' = {val} is out of range "
                f"[{meta['min']}, {meta['max']}]"
            )

        cleaned[feature] = float(val)

    return cleaned, None


# HELPER: PREPROCESSING PIPELINE
def preprocess_input(cleaned: dict) -> pd.DataFrame:
    """
    Apply the exact same preprocessing as training:
      1. Build DataFrame with original 10 features
      2. Scale using the saved StandardScaler
      3. Add 2 interaction features

    Returns:
        DataFrame with 12 features, ready for model.predict()
    """
    # Step 1: Build raw DataFrame (1 row)
    raw = pd.DataFrame([cleaned])[ORIGINAL_FEATURES]

    # Step 2: Scale (using training statistics — never refit the scaler!)
    scaled_arr = SCALER.transform(raw)
    scaled     = pd.DataFrame(scaled_arr, columns=ORIGINAL_FEATURES)

    # Step 3: Add interaction features
    scaled['study_efficiency'] = (
        scaled['total_study_hours'] * scaled['focus_ratio']
    )
    scaled['procrastination_penalty'] = (
        scaled['late_night_study_ratio'] /
        (scaled['days_until_deadline_avg'] + 0.1)
    )

    return scaled   # shape: (1, 12)



# HELPER: TIMING DECORATOR


def timed(f):
    """Decorator: adds 'processing_time_ms' to every JSON response."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start    = time.time()
        response = f(*args, **kwargs)
        elapsed  = round((time.time() - start) * 1000, 2)

        # Inject timing into response JSON
        if isinstance(response, tuple):
            data, code = response
        else:
            data, code = response, 200

        payload = data.get_json()
        payload['processing_time_ms'] = elapsed
        return jsonify(payload), code

    return wrapper



# ENDPOINT: GET /health


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    Used by: load balancers, monitoring tools, browser extension startup.

    Returns:
        200 OK with server status and model info
    """
    return jsonify({
        'success':           True,
        'status':            'healthy',
        'regressor':         artifacts['regressor_name'],
        'classifier':        artifacts['classifier_name'],
        'threshold':         round(THRESHOLD, 3),
        'features_expected': len(ORIGINAL_FEATURES),
        'pass_threshold':    60,
        'shap_mode':         'exact (TreeExplainer)' if SHAP_AVAILABLE else 'proxy (feature importance)',
    })



# ENDPOINT: GET /feature-info


@app.route('/feature-info', methods=['GET'])
def feature_info():
    """
    Returns metadata about every expected input feature.
    Used by: browser extension to know what data to collect and send.

    Returns:
        200 OK with feature names, ranges, units, and examples
    """
    return jsonify({
        'success':  True,
        'features': FEATURE_META,
        'count':    len(FEATURE_META),
    })



# ENDPOINT: POST /predict


@app.route('/predict', methods=['POST'])
@timed
def predict():
    """
    Main prediction endpoint.

    Expects JSON body with 10 raw feature values (see /feature-info).

    Returns:
        {
          success: true,
          prediction: {
            predicted_grade:   float,
            pass_probability:  float (0-100),
            risk_level:        'HIGH RISK' | 'MODERATE RISK' | 'ON TRACK',
            will_pass:         bool,
            recommendations: [
              {
                feature:  str,
                message:  str,
                priority: float  (SHAP magnitude)
              },
              ...
            ]
          },
          shap_contributions: { feature_name: shap_value, ... },
          processing_time_ms: float
        }
    """
    #  Parse JSON body 
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'success': False,
                        'error': 'Invalid JSON or missing Content-Type header'}), 400

    logger.info(f"POST /predict — incoming data: {data}")

    #  Validate 
    cleaned, error = validate_student_input(data)
    if error:
        logger.warning(f"Validation failed: {error}")
        return jsonify({'success': False, 'error': error}), 400

    #  Preprocess 
    X = preprocess_input(cleaned)

    #  Predict grade (regression) 
    predicted_grade   = float(REGRESSOR.predict(X)[0])
    predicted_grade   = round(np.clip(predicted_grade, 0, 100), 1)

    #  Predict pass probability (classification) 
    pass_probability  = float(CLASSIFIER.predict_proba(X)[0, 1])
    will_pass         = pass_probability >= THRESHOLD

    #  SHAP values for this student 
    # REG_EXPLAINER is the TreeExplainer loaded from explainability.pkl
    # It was fitted on the training set in Phase 4 — we reuse it here
    # so we don't re-fit on every request (expensive)
    if SHAP_AVAILABLE and REG_EXPLAINER is not None:
        # Exact SHAP via TreeExplainer — traces every tree path
        shap_vals = REG_EXPLAINER.shap_values(X)[0]   # shape: (12,)
    else:
        # Proxy: feature importance × standardised feature value
        # Direction is preserved (positive value → positive contribution)
        importances = REGRESSOR.feature_importances_
        shap_vals   = importances * X.values[0]

    shap_dict = {feat: round(float(val), 3)
                 for feat, val in zip(FEATURE_COLS, shap_vals)}

    #  Recommendations 
    result = generate_recommendations(
        student_features = X.iloc[0],
        shap_values_reg  = shap_vals,
        predicted_grade  = predicted_grade,
        pass_probability = pass_probability,
        threshold        = THRESHOLD,
        top_n            = 3,
    )

    # Clean up recommendations for JSON (remove non-serialisable items)
    recs_json = [
        {
            'feature':  r['feature'],
            'message':  r['message'],
            'priority': round(float(r['priority']), 3),
            'shap':     round(float(r['shap']), 3),
        }
        for r in result['recommendations']
    ]

    logger.info(
        f"Predicted grade={predicted_grade}, "
        f"pass_prob={pass_probability:.3f}, "
        f"risk={result['risk_level']}"
    )

    return jsonify({
        'success': True,
        'prediction': {
            'predicted_grade':   predicted_grade,
            'pass_probability':  round(pass_probability * 100, 1),
            'risk_level':        result['risk_level'],
            'will_pass':         bool(will_pass),
            'total_flags':       result['total_flags'],
            'recommendations':   recs_json,
        },
        'shap_contributions': shap_dict,
    })



# ENDPOINT: POST /predict/batch


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict for multiple students at once.
    More efficient than calling /predict N times separately.

    Expects JSON body:
        { "students": [ {student1_features}, {student2_features}, ... ] }

    Returns:
        { "success": true, "results": [ {prediction1}, {prediction2}, ... ] }
    """
    data = request.get_json(silent=True)
    if data is None or 'students' not in data:
        return jsonify({'success': False,
                        'error': "Body must be {'students': [...]}"}), 400

    students = data['students']
    if not isinstance(students, list) or len(students) == 0:
        return jsonify({'success': False,
                        'error': "'students' must be a non-empty list"}), 400

    if len(students) > 100:
        return jsonify({'success': False,
                        'error': "Batch size limited to 100 students"}), 400

    results  = []
    errors   = []

    for i, student_data in enumerate(students):
        cleaned, error = validate_student_input(student_data)
        if error:
            errors.append({'index': i, 'error': error})
            results.append(None)
            continue

        X               = preprocess_input(cleaned)
        predicted_grade = round(float(np.clip(REGRESSOR.predict(X)[0], 0, 100)), 1)
        pass_prob       = float(CLASSIFIER.predict_proba(X)[0, 1])
        will_pass       = pass_prob >= THRESHOLD

        risk = ('HIGH RISK' if pass_prob < THRESHOLD
                else 'MODERATE RISK' if pass_prob < THRESHOLD + 0.15
                else 'ON TRACK')

        results.append({
            'index':           i,
            'predicted_grade': predicted_grade,
            'pass_probability': round(pass_prob * 100, 1),
            'risk_level':      risk,
            'will_pass':       bool(will_pass),
        })

    logger.info(f"POST /predict/batch — {len(students)} students, "
                f"{len(errors)} errors")

    return jsonify({
        'success':       True,
        'total':         len(students),
        'succeeded':     len(students) - len(errors),
        'failed':        len(errors),
        'results':       results,
        'errors':        errors,
    })



# ERROR HANDLERS


@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False,
                    'error': f"Endpoint not found. "
                             f"Available: /health, /feature-info, /predict, /predict/batch"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'success': False,
                    'error': f"Method not allowed on this endpoint"}), 405


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'success': False,
                    'error': 'Internal server error. Check api.log for details.'}), 500



# RUN


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("PHASE 5: FLASK API STARTING")
    print("=" * 60)
    print(f"  Model:     {artifacts['regressor_name']} + {artifacts['classifier_name']}")
    print(f"  Threshold: {THRESHOLD:.3f}")
    print(f"  Endpoints:")
    print(f"    GET  http://localhost:5000/health")
    print(f"    GET  http://localhost:5000/feature-info")
    print(f"    POST http://localhost:5000/predict")
    print(f"    POST http://localhost:5000/predict/batch")
    print("=" * 60 + "\n")

    # debug=False in production, True during development
    # host='0.0.0.0' makes it accessible from other machines on the network
    app.run(host='0.0.0.0', port=5000, debug=True)