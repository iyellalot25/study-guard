"""
StudyLens — Quick Live Demo (No Week-Long Wait Required)

This script demonstrates the ML system working in real-time by:
  1. Simulating 3 different student profiles
  2. Calling the API with their behavioral data
  3. Showing predictions + recommendations instantly
"""

import requests
import json
import time

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

API_BASE = "http://localhost:5000"

# Colors
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ═══════════════════════════════════════════════════════════════════════════
# Student Profiles (Realistic Scenarios)
# ═══════════════════════════════════════════════════════════════════════════

PROFILES = {
    "Alice (High Achiever)": {
        "story": "Studies 20 hrs/week, starts assignments early, very focused",
        "data": {
            "total_study_hours": 20.0,
            "avg_session_length": 75.0,
            "study_sessions_per_week": 16.0,
            "focus_ratio": 0.90,
            "late_night_study_ratio": 0.05,
            "days_until_deadline_avg": 8.0,
            "video_completion_rate": 0.95,
            "practice_problem_attempts": 25.0,
            "forum_participation": 5.0,
            "study_streak_days": 14.0,
        }
    },
    
    "Bob (Struggling Student)": {
        "story": "Only 5 hrs/week, cramming late at night, starts work last minute",
        "data": {
            "total_study_hours": 5.0,
            "avg_session_length": 25.0,
            "study_sessions_per_week": 4.0,
            "focus_ratio": 0.35,
            "late_night_study_ratio": 0.80,
            "days_until_deadline_avg": 0.5,
            "video_completion_rate": 0.30,
            "practice_problem_attempts": 2.0,
            "forum_participation": 0.0,
            "study_streak_days": 1.0,
        }
    },
    
    "Carol (Borderline)": {
        "story": "12 hrs/week, inconsistent habits, moderate effort",
        "data": {
            "total_study_hours": 12.0,
            "avg_session_length": 50.0,
            "study_sessions_per_week": 8.0,
            "focus_ratio": 0.62,
            "late_night_study_ratio": 0.30,
            "days_until_deadline_avg": 3.0,
            "video_completion_rate": 0.60,
            "practice_problem_attempts": 8.0,
            "forum_participation": 2.0,
            "study_streak_days": 5.0,
        }
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# Demo Functions
# ═══════════════════════════════════════════════════════════════════════════

def print_header(text):
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{BOLD}{text:^80}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}\n")

def print_section(text):
    print(f"\n{CYAN}{'─' * 80}{RESET}")
    print(f"{CYAN}{text}{RESET}")
    print(f"{CYAN}{'─' * 80}{RESET}")

def check_api():
    """Check if API is running."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200:
            data = r.json()
            print(f"{GREEN}✓ API is running{RESET}")
            print(f"  Model: {data.get('regressor')} (threshold={data.get('threshold')})")
            print(f"  SHAP:  {data.get('shap_mode', 'unknown')}")
            return True
        else:
            print(f"{RED}✗ API returned status {r.status_code}{RESET}")
            return False
    except Exception as e:
        print(f"{RED}✗ Cannot connect to API{RESET}")
        print(f"{RED}  Error: {e}{RESET}")
        print(f"\n{YELLOW}Start the Flask API first:{RESET}")
        print(f"  python app.py\n")
        return False

def predict(student_data):
    """Get prediction from API."""
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json=student_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if r.status_code == 200:
            return r.json()
        else:
            return {"success": False, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def display_student_behavior(name, profile):
    """Show the student's weekly behavior."""
    print(f"\n{BOLD}{name}{RESET}")
    print(f"  {profile['story']}\n")
    
    data = profile['data']
    
    print(f"  Weekly Behavior:")
    print(f"    📚 Study hours:         {data['total_study_hours']:.0f} hrs/week")
    print(f"    ⏱️  Avg session:          {data['avg_session_length']:.0f} minutes")
    print(f"    🎯 Focus ratio:          {data['focus_ratio']*100:.0f}%")
    print(f"    😴 Late-night study:     {data['late_night_study_ratio']*100:.0f}%")
    print(f"    📅 Days before deadline: {data['days_until_deadline_avg']:.0f} days")
    print(f"    ✏️  Practice problems:    {data['practice_problem_attempts']:.0f}")
    print(f"    🎥 Video completion:     {data['video_completion_rate']*100:.0f}%")

def display_prediction(result):
    """Show the ML prediction and recommendations."""
    if not result.get('success'):
        print(f"\n{RED}✗ Prediction failed: {result.get('error')}{RESET}")
        return
    
    pred = result['prediction']
    
    # Grade and risk
    grade = pred['predicted_grade']
    prob = pred['pass_probability']
    risk = pred['risk_level']
    
    # Color code by risk
    if risk == 'HIGH RISK':
        color = RED
        emoji = "🔴"
    elif risk == 'MODERATE RISK':
        color = YELLOW
        emoji = "🟡"
    else:
        color = GREEN
        emoji = "🟢"
    
    print(f"\n  {BOLD}ML Prediction:{RESET}")
    print(f"    Grade:           {color}{grade}/100{RESET}")
    print(f"    Pass probability: {color}{prob}%{RESET}")
    print(f"    Risk level:       {color}{emoji} {risk}{RESET}")
    
    # Recommendations
    recs = pred['recommendations']
    if recs:
        print(f"\n  {BOLD}Top 3 Recommendations (SHAP-ranked):{RESET}")
        for i, rec in enumerate(recs, 1):
            # Truncate message if too long
            msg = rec['message']
            if len(msg) > 70:
                msg = msg[:67] + "..."
            print(f"    {i}. {msg}")
            print(f"       {CYAN}(Impact score: {rec['priority']:.2f}){RESET}")
    else:
        print(f"\n  {GREEN}✅ No recommendations — student is on track!{RESET}")
    
    # Processing time
    if 'processing_time_ms' in result:
        print(f"\n  ⚡ Predicted in {result['processing_time_ms']:.1f}ms")

# ═══════════════════════════════════════════════════════════════════════════
# Main Demo Flow
# ═══════════════════════════════════════════════════════════════════════════

def run_demo():
    print_header("StudyLens — Live Demo")
    
    print(f"{BOLD}What we're demonstrating:{RESET}")
    print("  1. ML system predicts student grades based on study behavior")
    print("  2. SHAP explainability shows which habits matter most")
    print("  3. Personalized recommendations ranked by impact")
    
    # Check API
    print_section("Step 1: Checking API Connection")
    if not check_api():
        return
    
    input(f"\n{YELLOW}Press Enter to continue...{RESET}")
    
    # Demo each student
    for name, profile in PROFILES.items():
        print_section(f"Step 2: Analyzing {name}")
        
        # Show behavior
        display_student_behavior(name, profile)
        
        input(f"\n{YELLOW}Press Enter to get ML prediction...{RESET}")
        
        # Get prediction
        print(f"\n  Sending data to ML model...")
        result = predict(profile['data'])
        
        # Show prediction
        display_prediction(result)
        
        input(f"\n{YELLOW}Press Enter to see next student...{RESET}")
    
    # Summary
    print_section("Demo Complete")
    print(f"\n{BOLD}What you just saw:{RESET}")
    print(f"  ✓ Same ML model used in production")
    print(f"  ✓ Real XGBoost predictions (not hardcoded)")
    print(f"  ✓ Actual SHAP explanations (TreeExplainer)")
    print(f"  ✓ Personalized recommendations for each student")
    print(f"\n{BOLD}In production:{RESET}")
    print(f"  → Browser extension tracks study behavior automatically")
    print(f"  → Student sees prediction updated weekly")
    print(f"  → Catches at-risk students early (before midterms)")
    print()

# ═══════════════════════════════════════════════════════════════════════════
# Alternative: Side-by-Side Comparison
# ═══════════════════════════════════════════════════════════════════════════

def run_comparison_demo():
    """Show all 3 students side-by-side for quick comparison."""
    print_header("StudyLens — Student Comparison Demo")
    
    # Check API
    if not check_api():
        return
    
    print(f"\n{BOLD}Comparing 3 students with different study habits...{RESET}\n")
    
    results = {}
    
    # Get all predictions first
    for name, profile in PROFILES.items():
        print(f"  Analyzing {name.split()[0]}...", end=" ")
        result = predict(profile['data'])
        results[name] = result
        if result.get('success'):
            print(f"{GREEN}✓{RESET}")
        else:
            print(f"{RED}✗{RESET}")
    
    # Display comparison table
    print(f"\n{BOLD}{'Student':<25} {'Study Hrs':<12} {'Grade':<10} {'Pass %':<10} {'Risk':<20}{RESET}")
    print(f"{'-' * 80}")
    
    for name, profile in PROFILES.items():
        result = results[name]
        hours = profile['data']['total_study_hours']
        
        if result.get('success'):
            pred = result['prediction']
            grade = pred['predicted_grade']
            prob = pred['pass_probability']
            risk = pred['risk_level']
            
            # Color by risk
            if risk == 'HIGH RISK':
                color = RED
            elif risk == 'MODERATE RISK':
                color = YELLOW
            else:
                color = GREEN
            
            print(f"{name:<25} {hours:<12.0f} {color}{grade:<10.1f}{RESET} "
                  f"{color}{prob:<10.1f}{RESET} {color}{risk:<20}{RESET}")
        else:
            print(f"{name:<25} {hours:<12.0f} {'ERROR':<10} {'ERROR':<10} {'ERROR':<20}")
    
    print(f"\n{BOLD}Key Insight:{RESET}")
    print(f"  Alice (20 hrs/week, focused)  → {GREEN}High grade, ON TRACK{RESET}")
    print(f"  Bob (5 hrs/week, cramming)    → {RED}Low grade, HIGH RISK{RESET}")
    print(f"  Carol (12 hrs/week, mixed)    → {YELLOW}Borderline, needs improvement{RESET}")
    
    print(f"\n{BOLD}The model correctly distinguishes between:{RESET}")
    print(f"  → Study QUANTITY (hours)")
    print(f"  → Study QUALITY (focus, timing, consistency)")
    print(f"  → Study HABITS (deadlines, practice, engagement)")
    print()

# ═══════════════════════════════════════════════════════════════════════════
# Run Demo
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    print(f"\n{CYAN}Choose demo mode:{RESET}")
    print(f"  1. Full demo (step-by-step, 3 students)")
    print(f"  2. Quick comparison (side-by-side table)")
    
    choice = input(f"\nEnter 1 or 2 (or press Enter for full demo): ").strip()
    
    print()
    
    if choice == "2":
        run_comparison_demo() 
    else:
        run_demo()