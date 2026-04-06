# StudyLens — ML-Powered Student Performance Prediction

**An end-to-end machine learning system that predicts student grades and provides personalized study recommendations using behavioral analytics.**

![Demo](https://img.shields.io/badge/demo-live-brightgreen) ![Python](https://img.shields.io/badge/python-3.8+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Problem Statement

70% of students who fail a course don't realize they're at risk until after midterms. By then, recovery is difficult.

**StudyLens** provides early warning by:
- Tracking ethical, privacy-respecting study behaviors
- Predicting likely grades using XGBoost
- Explaining predictions with SHAP
- Recommending high-impact interventions

---

## ✨ Key Features

### ML Pipeline
- **XGBoost Regression** (MAE: 7.90 points) — predicts final grade
- **XGBoost Classification** (AUC: 0.809) — predicts pass/fail probability
- **SHAP Explainability** — TreeExplainer for feature importance
- **Threshold Tuning** — optimized at 0.44 to catch 66% of at-risk students

### Browser Extension
- **Auto-tracking**: Study hours, focus ratio, session patterns
- **Manual inputs**: Practice problems, forum posts, deadlines, video completion
- **Dashboard**: Real-time predictions + top 3 SHAP-ranked recommendations
- **Privacy-first**: No URLs, no page content, no keystrokes — only aggregated signals

### Production-Ready API
- **Flask REST API** with 4 endpoints
- **Input validation** with detailed error messages
- **CORS-enabled** for cross-origin requests
- **Sub-10ms latency** (p95 < 20ms)

---

## 📊 Performance Metrics

| Model | Task | Metric | Value | Target |
|-------|------|--------|-------|--------|
| XGBoost Regressor | Grade Prediction | MAE | 7.90 pts | < 10 |
| XGBoost Regressor | Grade Prediction | R² | 0.396 | — |
| XGBoost Classifier | Pass/Fail | AUC-ROC | 0.809 | 0.85* |
| XGBoost Classifier | Pass/Fail | Recall (Fail) | 66% | > 50% |
| Flask API | Latency | p95 | < 20ms | < 100ms |

*AUC ceiling due to synthetic data's hidden `ability_factor` (irreducible error)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Browser Extension (Manifest V3)                            │
│    ├─ Content Script: Tracks study sessions (no PII)       │
│    ├─ Background Worker: Aggregates weekly data            │
│    └─ Popup UI: Displays predictions + recommendations     │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP POST /predict
┌─────────────────────────────────────────────────────────────┐
│  Flask API (Production Server)                              │
│    ├─ Input Validation: 10 features, range checks          │
│    ├─ Preprocessing: StandardScaler + feature engineering  │
│    ├─ Model Inference: XGBoost → grade + pass probability  │
│    ├─ SHAP Computation: TreeExplainer → feature impacts    │
│    └─ Recommendations: 11 rules ranked by SHAP magnitude   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  ML Artifacts (Pickled)                                     │
│    ├─ trained_models.pkl: XGBoost models + scaler          │
│    └─ explainability.pkl: SHAP TreeExplainer               │
└─────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. Extension tracks behavioral signals (time, focus, patterns)
2. Weekly aggregation → 10 features
3. API validates + scales + adds interaction features → 12 features
4. XGBoost predicts grade + pass probability
5. SHAP explains which behaviors matter most
6. Recommendations ranked by SHAP magnitude
7. Dashboard updates with predictions + advice

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Chrome browser
```

### Installation

```bash
# Clone repository
git clone https://github.com/iyellalot25/study-guard
cd studylens

# Install Python dependencies
pip install -r requirements.txt

# Start Flask API
python app.py
```

### Load Browser Extension

1. Open Chrome → `chrome://extensions/`
2. Enable "Developer mode" (top-right toggle)
3. Click "Load unpacked"
4. Select the `extension/` folder

### Run Demo

```bash
# Terminal 1: API running (from installation above)

# Terminal 2: Quick demo
python demo_live.py
```

---

## 📁 Project Structure

```
study-guard
├─ app.py
├─ demo.py
├─ explainability.pkl
├─ extension
│  ├─ background.js
│  ├─ content.js
│  ├─ icons
│  │  ├─ icon128.png
│  │  ├─ icon16.png
│  │  └─ icon48.png
│  ├─ manifest.json
│  ├─ popup.html
│  └─ popup.js
├─ feature_engineering_preprocessing.ipynb
├─ phase1.ipynb
├─ phase1v2+phase2.ipynb
├─ phase3.5.ipynb
├─ phase3.ipynb
├─ phase4.ipynb
├─ preprocessed_data.pkl
├─ README.md
├─ requirements.txt
├─ student_performance_data.csv
├─ test_api.py
└─ trained_models.pkl
```

---

## 🧪 Testing

### API Tests
```bash
# Start API first
python app.py

# Run test suite (18 tests, ~30 seconds)
python test_api.py
```

---

## 🔬 Technical Highlights

### 1. Synthetic Data Generation
- **5000 students** with realistic behavioral patterns
- **Hidden ability factor** (8-point Gaussian noise) creates irreducible error ceiling
- **80/20 pass/fail ratio** matches university statistics
- Prevents overfitting with noise injection

### 2. Feature Engineering
- **12 total features**: 10 base + 2 interactions
- `study_efficiency` = hours × focus_ratio
- `procrastination_penalty` = late_night_ratio / (deadline_buffer + 0.1)
- Standardized using training set statistics

### 3. Model Selection
- **XGBoost won over Random Forest** via GridSearchCV
- Hyperparameters: `max_depth=3`, `learning_rate=0.05` (conservative)
- Class imbalance handled with `scale_pos_weight=0.25`

### 4. Threshold Tuning
- Default threshold (0.50) caught only 46% of failing students
- Tuned threshold (0.44) catches 66% (+20pp improvement)
- Trade-off: 2% accuracy drop for 20% recall gain

### 5. Explainability
- **Real SHAP** (TreeExplainer, not approximation)
- Computed once during training, cached for inference
- 11 recommendation rules ranked by `|SHAP|` magnitude

### 6. Production Considerations
- Models in PKL, business logic in code (separation of concerns)
- Input validation prevents bad data from reaching model
- CORS + error handling for robust API
- Extension uses `chrome.storage.local` for persistence

---

## 📊 Feature Importance (SHAP)

**Top 5 Features by Mean |SHAP| (Regression):**

1. `total_study_hours` — 4.88 grade points
2. `focus_ratio` — 1.88 grade points
3. `late_night_study_ratio` — 1.79 grade points (negative)
4. `practice_problem_attempts` — 1.73 grade points
5. `avg_session_length` — 1.61 grade points

**Key Insight:** Study *quality* (focus, timing) matters as much as *quantity* (hours).

---

## 🛡️ Privacy & Ethics

- ✅ **No personal data**: No names, IDs, or demographics
- ✅ **No URL tracking**: Only domain categories (educational vs non-educational)
- ✅ **No content reading**: Never reads page text or keystrokes
- ✅ **Local storage**: Data stays in browser until sent to API
- ✅ **Opt-in**: Manual inputs for sensitive features (practice, deadlines)
- ✅ **Transparent**: User sees all tracked data in dashboard

---

## 🎓 Educational Use Cases

1. **Early Warning System**: Identify at-risk students before midterms
2. **Study Habit Optimization**: Show students which behaviors to change
3. **Intervention Prioritization**: Advisors focus on highest-risk students
4. **Self-Reflection Tool**: Students track their own progress weekly

---

## 🔮 Future Enhancements

### Short-term (Phase 9)
- [ ] LMS Integration (Canvas API) for automatic deadline tracking
- [ ] Site-specific practice problem detection (LeetCode, HackerRank)
- [ ] A/B test threshold values per course difficulty

### Medium-term
- [ ] Multi-course tracking (predict GPA, not just one course)
- [ ] Peer comparison (anonymized percentile ranking)
- [ ] Mobile app (iOS/Android)

### Long-term
- [ ] Real student data (IRB approval required)
- [ ] Causal inference (what happens if student follows advice?)
- [ ] Deep learning for temporal patterns (LSTM on session history)

---

**Built with:** Python • XGBoost • SHAP • Flask • JavaScript • Chrome Extensions API