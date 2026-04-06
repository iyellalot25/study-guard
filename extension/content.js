(function () {
  'use strict';

  // Educational domain list
  // We only collect data when the student is on an educational site.
  // We detect the domain category — never the specific URL/path.
  const EDUCATIONAL_DOMAINS = [
    'coursera.org', 'udemy.com', 'khanacademy.org', 'edx.org',
    'codecademy.com', 'freecodecamp.org', 'leetcode.com',
    'hackerrank.com', 'brilliant.org', 'skillshare.com',
    'linkedin.com/learning', 'pluralsight.com', 'egghead.io',
    'frontendmasters.com', 'youtube.com',   // educational content
    'stackoverflow.com', 'github.com',      // technical learning
    'medium.com', 'dev.to', 'docs.python.org', 'developer.mozilla.org',
    'w3schools.com', 'geeksforgeeks.org',
  ];

  const VIDEO_DOMAINS = [
    'youtube.com', 'coursera.org', 'udemy.com', 'edx.org',
    'linkedin.com/learning', 'pluralsight.com',
  ];

  const FORUM_DOMAINS = [
    'stackoverflow.com', 'reddit.com/r/learnprogramming',
    'github.com', 'dev.to',
  ];

  // Detect site type (no URL stored — only category)
  const hostname = window.location.hostname.replace('www.', '');

  const isEducational = EDUCATIONAL_DOMAINS.some(d => hostname.includes(d));
  const isVideo       = VIDEO_DOMAINS.some(d => hostname.includes(d));
  const isForum       = FORUM_DOMAINS.some(d => hostname.includes(d));

  // If not an educational site, do nothing at all
  if (!isEducational) return;

  // Session state (local to this tab's content script)
  const session = {
    startTime:      Date.now(),
    lastActiveTime: Date.now(),
    focusedMs:      0,         // ms spent with tab focused
    blurredMs:      0,         // ms spent with tab in background
    contextSwitches: 0,        // number of focus → blur events
    isLateNight:    isLateNight(),
    isVideo:        isVideo,
    isForum:        isForum,
    videoStarted:   false,
    postDetected:   false,
  };

  function isLateNight() {
    const h = new Date().getHours();
    return h >= 22 || h < 5;   // 10 PM – 5 AM
  }

  // Focus / blur tracking
  // These are the ONLY events we listen to.
  // We count transitions, not capture content.
  document.addEventListener('visibilitychange', () => {
    const now = Date.now();
    const elapsed = now - session.lastActiveTime;

    if (document.hidden) {
      // Tab went to background
      session.focusedMs += elapsed;
      session.contextSwitches += 1;
    } else {
      // Tab came back to foreground
      session.blurredMs += elapsed;
    }

    session.lastActiveTime = now;
  });

  // Video completion detection (heuristic)
  // We detect if a video element reached near-end.
  // We never capture what video was playing.
  if (isVideo) {
    const checkVideos = () => {
      const videos = document.querySelectorAll('video');
      videos.forEach(v => {
        if (!v._studylens_tracked) {
          v._studylens_tracked = true;
          v.addEventListener('play', () => {
            session.videoStarted = true;
          });
          v.addEventListener('timeupdate', () => {
            // "Completed" = watched past 85% of the video
            if (v.duration > 0 && (v.currentTime / v.duration) >= 0.85) {
              session.videoCompleted = true;
            }
          });
        }
      });
    };

    // Check now and again after a short delay (SPA pages load dynamically)
    setTimeout(checkVideos, 1000);
    setTimeout(checkVideos, 3000);
  }

  // Forum post detection (heuristic)
  // Detects if user submitted a form (likely a post/answer).
  // We never capture what they typed.
  if (isForum) {
    document.addEventListener('submit', () => {
      session.postDetected = true;
    }, { capture: true });
  }

  // Send aggregated signal to background worker on page unload
  const sendSessionData = () => {
    const now      = Date.now();
    const elapsed  = now - session.lastActiveTime;

    // Final accounting: was the tab focused or blurred at close?
    if (!document.hidden) {
      session.focusedMs += elapsed;
    } else {
      session.blurredMs += elapsed;
    }

    const totalMs    = session.focusedMs + session.blurredMs;
    const totalMins  = totalMs / 60000;

    // Only send if session was meaningful (> 1 minute)
    if (totalMins < 1) return;

    const focusRatio = session.focusedMs / (totalMs || 1);

    // This is ALL we send — aggregated numbers, no content
    const signal = {
      type:            'SESSION_END',
      sessionMinutes:  Math.round(totalMins),
      focusRatio:      Math.round(focusRatio * 1000) / 1000,
      contextSwitches: session.contextSwitches,
      isLateNight:     session.isLateNight,
      isVideo:         session.isVideo,
      videoStarted:    session.videoStarted  || false,
      videoCompleted:  session.videoCompleted || false,
      isForum:         session.isForum,
      postDetected:    session.postDetected  || false,
      timestamp:       now,
    };

    // Chrome message passing — sends to background.js
    chrome.runtime.sendMessage(signal).catch(() => {
      // Extension may have been reloaded — silently ignore
    });
  };

  // Send on page unload (beforeunload is unreliable in MV3, use pagehide)
  window.addEventListener('pagehide', sendSessionData);

  // Also send a periodic heartbeat every 5 minutes for long sessions
  const heartbeatInterval = setInterval(() => {
    const minutesElapsed = (Date.now() - session.startTime) / 60000;
    if (minutesElapsed >= 5) {
      sendSessionData();
      // Reset start time so we don't double-count
      session.startTime      = Date.now();
      session.lastActiveTime = Date.now();
      session.focusedMs      = 0;
      session.blurredMs      = 0;
      session.contextSwitches = 0;
    }
  }, 5 * 60 * 1000);

  // Clean up interval if extension context is invalidated
  window.addEventListener('pagehide', () => clearInterval(heartbeatInterval));

})();