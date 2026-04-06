'use strict';

const API_BASE    = 'http://localhost:5000';
const STORAGE_KEY = 'studylens_weekly_data';
const RESULT_KEY  = 'studylens_last_prediction';
const ALARM_NAME  = 'studylens_hourly_predict';

// ─────────────────────────────────────────────────────────────
// Default weekly accumulator structure
// Resets every Monday at midnight
// ─────────────────────────────────────────────────────────────
function defaultWeeklyData() {
  return {
    weekStart:              getMondayTimestamp(),

    // Raw accumulators (aggregated from session signals)
    totalStudyMinutes:      0,   // → total_study_hours
    sessionCount:           0,   // → study_sessions_per_week
    sessionLengthsMin:      [],  // → avg_session_length
    focusRatios:            [],  // → focus_ratio
    lateNightMinutes:       0,   // → late_night_study_ratio
    totalActiveMinutes:     0,   // denominator for late night ratio
    studyDays:              new Set(),  // days with any study → streak
    lastStudyDay:           null,
    streakDays:             0,

    // Manual user inputs (from popup)
    manualPracticeCount:    0,   // user-logged practice problems
    manualForumPosts:       0,   // user-logged forum posts
    manualDeadlineBuffer:   5,   // user-logged avg days before deadline (default 5)
    manualVideoCompletion:  0.5, // user-logged video completion rate (default 0.5)
  };
}

function getMondayTimestamp() {
  const now  = new Date();
  const day  = now.getDay();                    // 0=Sun, 1=Mon...
  const diff = (day === 0 ? -6 : 1 - day);     // days back to Monday
  const mon  = new Date(now);
  mon.setDate(now.getDate() + diff);
  mon.setHours(0, 0, 0, 0);
  return mon.getTime();
}

// ─────────────────────────────────────────────────────────────
// Load / save weekly data from chrome.storage.local
// ─────────────────────────────────────────────────────────────
async function loadWeeklyData() {
  return new Promise(resolve => {
    chrome.storage.local.get([STORAGE_KEY], result => {
      const stored = result[STORAGE_KEY];
      if (!stored || stored.weekStart < getMondayTimestamp()) {
        // New week — reset accumulator
        console.log('[StudyLens] New week — resetting accumulator');
        resolve(defaultWeeklyData());
      } else {
        // Restore Set (serialised as array in storage)
        if (Array.isArray(stored.studyDays)) {
          stored.studyDays = new Set(stored.studyDays);
        }
        resolve(stored);
      }
    });
  });
}

async function saveWeeklyData(data) {
  // Convert Set → Array for JSON serialisation
  const toStore = { ...data, studyDays: [...data.studyDays] };
  return new Promise(resolve => {
    chrome.storage.local.set({ [STORAGE_KEY]: toStore }, resolve);
  });
}

// ─────────────────────────────────────────────────────────────
// Compute features from weekly accumulator
// This is where raw aggregates become the 10 model features
// ─────────────────────────────────────────────────────────────
function computeFeatures(data) {
  const totalStudyHours = data.totalStudyMinutes / 60;
  const avgSessionLen   = data.sessionLengthsMin.length > 0
    ? data.sessionLengthsMin.reduce((a, b) => a + b, 0) / data.sessionLengthsMin.length
    : 30;   // default if no sessions yet

  const avgFocusRatio   = data.focusRatios.length > 0
    ? data.focusRatios.reduce((a, b) => a + b, 0) / data.focusRatios.length
    : 0.5;

  const lateNightRatio  = data.totalActiveMinutes > 0
    ? data.lateNightMinutes / data.totalActiveMinutes
    : 0;

  // Manual inputs (user logs these in popup)
  const videoCompletionRate = data.manualVideoCompletion || 0.5;
  const practiceAttempts    = data.manualPracticeCount || 0;
  const forumPosts          = data.manualForumPosts || 0;
  const daysUntilDeadlineAvg = data.manualDeadlineBuffer || 5;  // single value, not average

  // Streak: compute from studyDays set
  const studyDays = data.studyDays || new Set();
  let streak = 0;
  let check  = new Date();
  while (studyDays.has(check.toDateString())) {
    streak++;
    check.setDate(check.getDate() - 1);
  }

  // Clamp all features to their valid API ranges
  return {
    total_study_hours:          Math.min(Math.max(totalStudyHours, 0), 80),
    avg_session_length:         Math.min(Math.max(avgSessionLen, 5), 300),
    study_sessions_per_week:    Math.min(Math.max(data.sessionCount || 0, 0), 50),
    focus_ratio:                Math.min(Math.max(avgFocusRatio, 0), 1),
    late_night_study_ratio:     Math.min(Math.max(lateNightRatio, 0), 1),
    days_until_deadline_avg:    Math.min(Math.max(daysUntilDeadlineAvg, 0), 30),
    video_completion_rate:      Math.min(Math.max(videoCompletionRate, 0), 1),
    practice_problem_attempts:  Math.min(Math.max(practiceAttempts, 0), 200),
    forum_participation:        Math.min(Math.max(forumPosts, 0), 50),
    study_streak_days:          Math.min(Math.max(streak, 0), 365),
  };
}

// ─────────────────────────────────────────────────────────────
// Call Flask API for prediction
// ─────────────────────────────────────────────────────────────
async function fetchPrediction(features) {
  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(features),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || `HTTP ${response.status}`);
    }

    return await response.json();

  } catch (err) {
    console.error('[StudyLens] API error:', err.message);
    return null;
  }
}

// ─────────────────────────────────────────────────────────────
// Save prediction result + timestamp
// ─────────────────────────────────────────────────────────────
async function savePredictionResult(result, features) {
  return new Promise(resolve => {
    chrome.storage.local.set({
      [RESULT_KEY]: {
        prediction: result.prediction,
        features:   features,
        fetchedAt:  Date.now(),
      }
    }, resolve);
  });
}

// ─────────────────────────────────────────────────────────────
// Main prediction pipeline
// ─────────────────────────────────────────────────────────────
async function runPrediction() {
  console.log('[StudyLens] Running prediction pipeline...');

  const weeklyData = await loadWeeklyData();
  const features   = computeFeatures(weeklyData);

  console.log('[StudyLens] Computed features:', features);

  const result = await fetchPrediction(features);

  if (result && result.success) {
    await savePredictionResult(result, features);

    const pred = result.prediction;
    console.log(
      `[StudyLens] Prediction: grade=${pred.predicted_grade}, ` +
      `pass=${pred.pass_probability}%, risk=${pred.risk_level}`
    );

    // Show notification if student is at risk
    if (pred.risk_level === 'HIGH RISK') {
      chrome.notifications.create('studylens_alert', {
        type:    'basic',
        iconUrl: 'icons/icon48.png',
        title:   '⚠️ StudyLens Alert',
        message: `Predicted grade: ${pred.predicted_grade}/100. ` +
                 `You may be at risk of failing. Click the extension to see recommendations.`,
      });
    }
  } else {
    console.warn('[StudyLens] Prediction failed or API unavailable');
  }
}

// ─────────────────────────────────────────────────────────────
// Message listener — receives signals from content.js
// ─────────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

  if (message.type === 'SESSION_END') {
    handleSessionEnd(message).then(() => sendResponse({ ok: true }));
    return true;  // keep message channel open for async response
  }

  if (message.type === 'GET_STATUS') {
    // Popup is asking for latest data
    chrome.storage.local.get([RESULT_KEY, STORAGE_KEY], async result => {
      const weekly  = result[STORAGE_KEY] || defaultWeeklyData();
      if (Array.isArray(weekly.studyDays)) {
        weekly.studyDays = new Set(weekly.studyDays);
      }
      sendResponse({
        prediction: result[RESULT_KEY] || null,
        features:   computeFeatures(weekly),
        weeklyData: {
          totalStudyHours:       weekly.totalStudyMinutes / 60,
          sessionCount:          weekly.sessionCount,
          manualPracticeCount:   weekly.manualPracticeCount || 0,
          manualForumPosts:      weekly.manualForumPosts || 0,
          manualVideoCompletion: weekly.manualVideoCompletion || 0.5,
          manualDeadlineBuffer:  weekly.manualDeadlineBuffer || 5,
        },
      });
    });
    return true;
  }

  if (message.type === 'SET_DEADLINE') {
    // Popup sends deadline info (days until assignment due)
    handleSetDeadline(message.daysUntil).then(() => sendResponse({ ok: true }));
    return true;
  }

  if (message.type === 'LOG_PRACTICE') {
    // Popup manually logs practice problems
    handleLogPractice(message.count).then(() => sendResponse({ ok: true }));
    return true;
  }

  if (message.type === 'LOG_FORUM') {
    // Popup manually logs forum posts
    handleLogForum(message.count).then(() => sendResponse({ ok: true }));
    return true;
  }

  if (message.type === 'SET_VIDEO_COMPLETION') {
    // Popup sets video completion ratio
    handleSetVideoCompletion(message.ratio).then(() => sendResponse({ ok: true }));
    return true;
  }

  if (message.type === 'RESET_DATA') {
    chrome.storage.local.remove([STORAGE_KEY, RESULT_KEY], () => {
      sendResponse({ ok: true });
    });
    return true;
  }
});

// ─────────────────────────────────────────────────────────────
// Handle session end signal from content.js
// ─────────────────────────────────────────────────────────────
async function handleSessionEnd(signal) {
  const data = await loadWeeklyData();

  data.totalStudyMinutes  += signal.sessionMinutes;
  data.totalActiveMinutes += signal.sessionMinutes;
  data.sessionCount       += 1;
  data.sessionLengthsMin.push(signal.sessionMinutes);
  data.focusRatios.push(signal.focusRatio);

  if (signal.isLateNight) {
    data.lateNightMinutes += signal.sessionMinutes;
  }

  if (signal.isVideo && signal.videoStarted)   data.videosStarted   += 1;
  if (signal.isVideo && signal.videoCompleted) data.videosCompleted += 1;
  if (signal.isForum && signal.postDetected)   data.forumPosts      += 1;

  // Mark today as a study day (for streak calculation)
  data.studyDays.add(new Date().toDateString());
  data.lastStudyDay = new Date().toDateString();

  await saveWeeklyData(data);
  console.log(`[StudyLens] Session saved: ${signal.sessionMinutes}min, focus=${signal.focusRatio}`);
}

async function handleSetDeadline(daysUntil) {
  const data = await loadWeeklyData();
  data.manualDeadlineBuffer = Number(daysUntil) || 5;
  await saveWeeklyData(data);
  console.log(`[StudyLens] Deadline buffer set to ${daysUntil} days`);
}

async function handleLogPractice(count) {
  const data = await loadWeeklyData();
  data.manualPracticeCount = Number(count) || 0;
  await saveWeeklyData(data);
}

async function handleLogForum(count) {
  const data = await loadWeeklyData();
  data.manualForumPosts = Number(count) || 0;
  await saveWeeklyData(data);
}

async function handleSetVideoCompletion(ratio) {
  const data = await loadWeeklyData();
  data.manualVideoCompletion = Math.min(Math.max(Number(ratio) || 0.5, 0), 1);
  await saveWeeklyData(data);
}

// ─────────────────────────────────────────────────────────────
// Alarms — run prediction pipeline hourly
// ─────────────────────────────────────────────────────────────
chrome.alarms.create(ALARM_NAME, { periodInMinutes: 60 });

chrome.alarms.onAlarm.addListener(alarm => {
  if (alarm.name === ALARM_NAME) runPrediction();
});

// Also run once immediately on extension startup
chrome.runtime.onInstalled.addListener(() => {
  console.log('[StudyLens] Extension installed — running initial prediction');
  runPrediction();
});

chrome.runtime.onStartup.addListener(() => {
  console.log('[StudyLens] Browser started — running prediction');
  runPrediction();
});