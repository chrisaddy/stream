#!/bin/bash
# Drive the demo walkthrough via Chrome AppleScript while ffmpeg records.
# Each section scrolls to the relevant area and pauses for narration timing.

JS() {
  osascript -e "
tell application \"Google Chrome\"
  tell active tab of front window
    execute javascript \"$1\"
  end tell
end tell" 2>/dev/null
}

echo "=== Section 1: Intro (0-22s) ==="
JS "window.scrollTo(0, 0)"
sleep 22

echo "=== Section 2: Live Scoring (22-47s) ==="
JS "window.scrollTo({top: 300, behavior: 'smooth'})"
sleep 8
JS "window.scrollTo({top: 600, behavior: 'smooth'})"
sleep 8
JS "window.scrollTo({top: 800, behavior: 'smooth'})"
sleep 9

echo "=== Section 3: Alert Generation (47-67s) ==="
# Lower threshold
JS "var s=document.getElementById('threshold-slider'); if(s){s.value='0.15'; document.getElementById('threshold-value').textContent='0.15'; s.dispatchEvent(new Event('change',{bubbles:true}));} 'done'"
sleep 5
JS "window.scrollTo({top: 900, behavior: 'smooth'})"
sleep 15

echo "=== Section 4: Instant Learning (67-107s) ==="
# Click first review button
JS "var links=document.querySelectorAll('#alert-feed a'); var r=Array.from(links).filter(function(a){return a.textContent.trim()==='Review'}); if(r.length>0){r[0].click();} 'clicked'"
sleep 8
# Mark as TP
JS "var btns=document.querySelectorAll('#alert-detail button'); var tp=Array.from(btns).find(function(b){return b.textContent.toLowerCase().includes('true positive')}); if(tp){tp.click();} 'tp'"
sleep 5
# Click second review
JS "var links=document.querySelectorAll('#alert-feed a'); var r=Array.from(links).filter(function(a){return a.textContent.trim()==='Review'}); if(r.length>1){r[1].click();} else if(r.length>0){r[0].click();} 'clicked2'"
sleep 5
# Mark as FP
JS "var btns=document.querySelectorAll('#alert-detail button'); var fp=Array.from(btns).find(function(b){return b.textContent.toLowerCase().includes('false positive')}); if(fp){fp.click();} 'fp'"
sleep 5
# Click third review + TP
JS "var links=document.querySelectorAll('#alert-feed a'); var r=Array.from(links).filter(function(a){return a.textContent.trim()==='Review'}); if(r.length>0){r[0].click();} 'clicked3'"
sleep 5
JS "var btns=document.querySelectorAll('#alert-detail button'); var tp=Array.from(btns).find(function(b){return b.textContent.toLowerCase().includes('true positive')}); if(tp){tp.click();} 'tp2'"
sleep 12

echo "=== Section 5: Metrics & Race (107-141s) ==="
JS "window.scrollTo({top: 1200, behavior: 'smooth'})"
sleep 12
JS "window.scrollTo({top: 1500, behavior: 'smooth'})"
sleep 10
JS "window.scrollTo({top: 1800, behavior: 'smooth'})"
sleep 12

echo "=== Section 6: Drift Detection (141-164s) ==="
JS "window.scrollTo({top: 1400, behavior: 'smooth'})"
sleep 23

echo "=== Section 7: Self-Healing (164-189s) ==="
# Trigger force adaptation via API
JS "fetch('/api/v1/online/adaptation/force',{method:'POST'}).then(function(r){return r.json()}).then(function(d){console.log(d)}); 'force'"
sleep 25

echo "=== Section 8: Anomaly Detection (189-209s) ==="
JS "window.scrollTo({top: 1700, behavior: 'smooth'})"
sleep 20

echo "=== Section 9: Outro (209-233s) ==="
JS "window.scrollTo({top: 900, behavior: 'smooth'})"
sleep 8
JS "window.scrollTo({top: 300, behavior: 'smooth'})"
sleep 8
JS "window.scrollTo({top: 0, behavior: 'smooth'})"
sleep 8

echo "=== Demo complete ==="
