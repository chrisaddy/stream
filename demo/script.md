# STREAM Demo Video Script

Voice: Chris (iP95p4xoKVk53GoZ742B) — v2 settings
Total target: ~4-5 minutes

---

## Section 1: INTRO (0:00 - 0:25)
**Screen: Title card, then app landing page**

> This is Stream — a real-time Bitcoin compliance platform. It pulls live transactions from the Bitcoin mempool, scores each one for illicit activity risk, and flags suspicious transactions for analyst review. But what makes it different is what happens after the analyst reviews — the system learns instantly and adapts in real-time. Let me show you.

## Section 2: LIVE SCORING FEED (0:25 - 1:00)
**Screen: Home page with oscilloscope and scoring feed streaming**

> Right now you're looking at live Bitcoin transactions flowing in from the mempool. Each one gets scored by our ML model — you can see the risk score, the fee rate, transaction size, and an anomaly score from our unsupervised detector. The oscilloscope at the top traces both the risk scores in cyan and anomaly scores in purple. Everything you see is real mempool data, scored in real-time.

## Section 3: ALERT GENERATION (1:00 - 1:30)
**Screen: Lower the threshold slider, watch alerts appear**

> The risk threshold controls what gets flagged. I'm going to lower it now so we can trigger some alerts for the demo. Watch the alert queue on the right — as transactions cross the threshold, they get flagged automatically for compliance review. Each alert includes the transaction details, the risk score, and a SHAP explanation showing which features drove the score.

## Section 4: THE REVIEW — INSTANT LEARNING (1:30 - 2:30)
**Screen: Click an alert, review as TP, then FP. Watch model badge change.**

> Here's the key moment. I'm going to review this alert as a true positive — meaning yes, this looks suspicious. Watch what happens behind the scenes. The online model just learned from that label. The metrics panel updated. The model race started tracking. Now I'll review another one as a false positive. And now — look at the scoring feed. See the model badge? It switched from "heuristic" to "M L". The online model has seen both classes and activated. Every new transaction is now being scored by a model that learned from our analyst feedback, with zero retraining delay.

## Section 5: ONLINE METRICS AND MODEL RACE (2:30 - 3:15)
**Screen: Pan to online metrics panel, then model race standings and convergence chart**

> Down here, the online metrics panel shows live precision, recall, F1, and ROC AUC — both cumulative and rolling window. These update with every review. And next to it, the model race — three different classifiers competing in real-time: Logistic Regression, Hoeffding Tree, and Gaussian Naive Bayes. The standings table shows who's winning on rolling F1, and the convergence chart tracks how each model improves over time. This is something you simply cannot do with batch ML.

## Section 6: DRIFT DETECTION (3:15 - 3:50)
**Screen: Drift monitor panel showing PSI + ADWIN status**

> The drift monitor runs two detectors simultaneously. PSI tracks the overall score distribution against a baseline — green means stable, red means the model's behavior has shifted. ADWIN provides adaptive windowing for change-point detection, catching sudden distribution shifts that PSI might miss. Together, they give us early warning when the mempool environment changes.

## Section 7: SELF-HEALING ADAPTATION (3:50 - 4:25)
**Screen: Click FORCE ADAPTATION button, watch state transitions**

> Now watch this. I'm hitting the Force Adaptation button. The system transitions through its state machine — Adapting, Stabilizing, then back to Stable. What just happened? It reset the drift detectors, cleared the online metrics, and restarted the model race. The entire ML pipeline healed itself. In production, this fires automatically after three consecutive drift signals — no human intervention needed.

## Section 8: ANOMALY DETECTION (4:25 - 4:45)
**Screen: Anomaly detector panel, highlight high anomaly scores in feed**

> The anomaly detector runs independently using Half-Space Trees — it's completely unsupervised. It learns the normal pattern of transactions and flags outliers. After its calibration period, you'll see anomaly scores appear in the feed. High scores get highlighted in red. This catches novel threats that the supervised model hasn't been trained on yet.

## Section 9: OUTRO (4:45 - 5:00)
**Screen: Pull back to full dashboard view**

> That's Stream. Live scoring, instant learning, autonomous drift detection, self-healing adaptation, and unsupervised anomaly detection — all running on real Bitcoin mempool data. This isn't a batch pipeline that retrains overnight. It's a living ML system that gets smarter with every analyst decision.
