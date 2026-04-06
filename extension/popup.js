'use strict';

const API_BASE = 'http://localhost:5000';

// Utility helpers

function $(id) { return document.getElementById(id); }

function sendToBackground(message) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, response => {
      if (chrome.runtime.lastError) {
        reject(chrome.runtime.lastError);
      } else {
        resolve(response);
      }
    });
  });
}

function timeAgo(timestamp) {
  if (!timestamp) return 'Never updated';
  const mins = Math.round((Date.now() - timestamp) / 60000);
  if (mins < 1)  return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  return hrs < 24 ? `${hrs}h ago` : `${Math.floor(hrs / 24)}d ago`;
}

function clamp(val, lo, hi) { return Math.max(lo, Math.min(hi, val)); }

// API health check

async function checkApiHealth() {
  const banner = $('statusBanner');
  const text   = $('statusText');

  try {
    const res  = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();

    if (data.success) {
      banner.className = 'status-banner ok';
      text.textContent = `API connected · ${data.regressor} · threshold ${data.threshold}`;
      return true;
    }
  } catch {
    // fall through to error state
  }

  banner.className = 'status-banner error';
  text.textContent = 'API offline — start Flask server: python app.py';
  return false;
}

// Render grade circle + risk badge

function renderGrade(prediction) {
  if (!prediction) {
    $('gradeNumber').textContent = '--';
    $('passProbText').textContent = 'No prediction yet';
    $('riskBadge').textContent = 'PENDING';
    $('riskBadge').className = 'risk-badge';
    $('updatedAt').textContent = 'Click Refresh to run first prediction';
    return;
  }

  const pred = prediction.prediction;
  const riskClass = {
    'HIGH RISK':      'high-risk',
    'MODERATE RISK':  'moderate',
    'ON TRACK':       'on-track',
  }[pred.risk_level] || 'on-track';

  $('gradeNumber').textContent = pred.predicted_grade;
  $('gradeCircle').className   = `grade-circle ${riskClass}`;
  $('passProbText').textContent = `Pass probability: ${pred.pass_probability}%`;
  $('riskBadge').textContent   = pred.risk_level;
  $('riskBadge').className     = `risk-badge ${riskClass}`;
  $('updatedAt').textContent   = `Updated ${timeAgo(prediction.fetchedAt)}`;
}

// Render feature bars

// Display config for each feature:
// [icon, label, min, max, higherIsBetter]
const FEATURE_CONFIG = {
  total_study_hours:          ['📚', 'Study hours/week',   0,   40,  true],
  avg_session_length:         ['⏱️',  'Avg session (min)',  5,  180,  true],
  study_sessions_per_week:    ['📅', 'Sessions/week',      0,   25,  true],
  focus_ratio:                ['🎯', 'Focus ratio',        0,    1,  true],
  late_night_study_ratio:     ['😴', 'Late-night ratio',   0,    1,  false],  // lower = better
  days_until_deadline_avg:    ['⏰', 'Days until deadline', 0,  14,  true],
  video_completion_rate:      ['🎥', 'Video completion',   0,    1,  true],
  practice_problem_attempts:  ['✏️',  'Practice attempts',  0,   30,  true],
  forum_participation:        ['💬', 'Forum posts/week',   0,   10,  true],
  study_streak_days:          ['🔥', 'Study streak (days)', 0,  30,  true],
};

function renderFeatures(features) {
  const list = $('featureList');
  list.innerHTML = '';

  for (const [key, [icon, label, lo, hi, higherBetter]] of Object.entries(FEATURE_CONFIG)) {
    const val     = features[key] ?? 0;
    const pct     = clamp((val - lo) / (hi - lo), 0, 1) * 100;
    const goodPct = higherBetter ? pct : 100 - pct;

    const barClass = goodPct >= 66 ? 'high' : goodPct >= 33 ? 'medium' : 'low';

    // Format display value
    let display;
    if (key === 'focus_ratio' || key === 'late_night_study_ratio' || key === 'video_completion_rate') {
      display = `${Math.round(val * 100)}%`;
    } else if (key === 'total_study_hours') {
      display = `${val.toFixed(1)}h`;
    } else {
      display = Number.isInteger(val) ? val : val.toFixed(1);
    }

    const row = document.createElement('div');
    row.className = 'feature-row';
    row.innerHTML = `
      <span class="feature-icon">${icon}</span>
      <span class="feature-name">${label}</span>
      <div class="feature-bar-wrap">
        <div class="feature-bar ${barClass}" style="width: ${pct}%"></div>
      </div>
      <span class="feature-val">${display}</span>
    `;
    list.appendChild(row);
  }
}

// Render recommendations

function renderRecommendations(prediction) {
  const list = $('recList');
  list.innerHTML = '';

  if (!prediction) {
    list.innerHTML = '<div class="no-recs" style="color:#4b5563">Run a prediction first</div>';
    return;
  }

  const recs = prediction.prediction.recommendations;

  if (!recs || recs.length === 0) {
    list.innerHTML = '<div class="no-recs">✅ No issues detected — keep it up!</div>';
    return;
  }

  recs.forEach((rec, i) => {
    const impactClass = i === 0 ? 'high-impact' : i === 1 ? 'med-impact' : '';
    const item = document.createElement('div');
    item.className = `rec-item ${impactClass}`;
    item.innerHTML = `
      <div>${rec.message}</div>
      <div class="rec-priority">Impact score: ${Math.abs(rec.priority).toFixed(2)}</div>
    `;
    list.appendChild(item);
  });
}

// Trigger a fresh prediction via background.js → Flask API

async function triggerPrediction() {
  const btn = $('refreshBtn');
  btn.textContent = '⏳ Predicting...';
  btn.disabled = true;

  try {
    // Ask background worker to call the API and save results
    // background.js handles this via runPrediction()
    // We send a special message to invoke it synchronously for the popup
    const status = await sendToBackground({ type: 'GET_STATUS' });

    if (status && status.features) {
      // Call the API directly from popup (background may not respond fast enough)
      const apiAlive = await checkApiHealth();
      if (!apiAlive) return;

      const res = await fetch(`${API_BASE}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(status.features),
      });

      const data = await res.json();

      if (data.success) {
        // Save result to storage so background.js and future popups see it
        await new Promise(resolve => {
          chrome.storage.local.set({
            'studylens_last_prediction': {
              prediction: data.prediction,
              features:   status.features,
              fetchedAt:  Date.now(),
            }
          }, resolve);
        });

        renderGrade({ prediction: data.prediction, fetchedAt: Date.now() });
        renderRecommendations({ prediction: data.prediction });
      }
    }
  } catch (err) {
    console.error('[StudyLens popup] Prediction error:', err);
    $('statusBanner').className = 'status-banner error';
    $('statusText').textContent = `Error: ${err.message}`;
  } finally {
    btn.textContent = '↻ Refresh';
    btn.disabled    = false;
  }
}

// Load and render everything

async function loadAndRender() {
  // 1. Check API health
  await checkApiHealth();

  // 2. Ask background for status
  let status;
  try {
    status = await sendToBackground({ type: 'GET_STATUS' });
  } catch (err) {
    console.warn('[StudyLens popup] Background unavailable:', err.message);
    status = null;
  }

  // 3. Get saved prediction from storage directly (fallback)
  const stored = await new Promise(resolve => {
    chrome.storage.local.get(['studylens_last_prediction'], r => resolve(r));
  });

  const prediction = stored['studylens_last_prediction'] || null;
  const features   = status?.features || {};

  // 4. Render
  renderGrade(prediction);
  renderFeatures(features);
  renderRecommendations(prediction);

  // 5. Show content, hide spinner
  $('loadingView').style.display  = 'none';
  $('mainContent').style.display  = 'block';
}

// Event listeners

document.addEventListener('DOMContentLoaded', () => {

  // Refresh button — trigger a new prediction
  $('refreshBtn').addEventListener('click', async () => {
    await triggerPrediction();
  });

  // Reset button — wipe all stored data
  $('resetBtn').addEventListener('click', async () => {
    if (!confirm('Reset all study data for this week?')) return;
    await sendToBackground({ type: 'RESET_DATA' });
    await loadAndRender();
  });

  // Log practice problems (sets total for the week)
  $('practiceBtn').addEventListener('click', async () => {
    const val = parseInt($('practiceInput').value);
    if (isNaN(val) || val < 0 || val > 200) {
      alert('Please enter a number between 0 and 200');
      return;
    }
    await sendToBackground({ type: 'LOG_PRACTICE', count: val });
    $('practiceInput').value = '';
    const status = await sendToBackground({ type: 'GET_STATUS' });
    if (status?.features) renderFeatures(status.features);
  });

  // Log forum posts (sets total for the week)
  $('forumBtn').addEventListener('click', async () => {
    const val = parseInt($('forumInput').value);
    if (isNaN(val) || val < 0 || val > 50) {
      alert('Please enter a number between 0 and 50');
      return;
    }
    await sendToBackground({ type: 'LOG_FORUM', count: val });
    $('forumInput').value = '';
    const status = await sendToBackground({ type: 'GET_STATUS' });
    if (status?.features) renderFeatures(status.features);
  });

  // Log deadline (adds to list, computes average)
  $('deadlineBtn').addEventListener('click', async () => {
    const val = parseFloat($('deadlineInput').value);
    if (isNaN(val) || val < 0 || val > 30) {
      alert('Please enter a number between 0 and 30');
      return;
    }
    await sendToBackground({ type: 'SET_DEADLINE', daysUntil: val });
    $('deadlineInput').value = '';
    const status = await sendToBackground({ type: 'GET_STATUS' });
    if (status?.features) renderFeatures(status.features);
  });

  // Set video completion percentage
  $('videoBtn').addEventListener('click', async () => {
    const val = parseInt($('videoInput').value);
    if (isNaN(val) || val < 0 || val > 100) {
      alert('Please enter a number between 0 and 100');
      return;
    }
    await sendToBackground({ type: 'SET_VIDEO_COMPLETION', ratio: val / 100 });
    $('videoInput').value = '';
    const status = await sendToBackground({ type: 'GET_STATUS' });
    if (status?.features) renderFeatures(status.features);
  });

  // Initial load
  loadAndRender();
});