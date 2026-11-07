# AI Referee

Browser-based martial-arts referee. Punches, kicks and knee strikes are judged by **OpenAI CLIP** on a Python backend; throws, falls and holding are tracked in the browser. Presets: Boxing, Judo, Kickboxing.

```
camera(s) / video file          (up to 3 cameras, each with its own pose tracker)
   │
   ▼  MediaPipe Pose (2 people, in the browser)
 pose tracker ──► candidate punch / kick / knee? ──► 0.8 s clip (≤16 JPEG frames + skeletons) ──► POST /api/judge
                                                                                        │
                                              CLIP backend (backend/app) ◄───────────────┘
                                              geometry + OpenAI CLIP zero-shot on crops
                                                                                        │ verdict: 7 answers
 ruleset (boxing / judo / kickboxing) ◄────────────────────────────────────────────────────────┘
   ▼
 scoreboard · call log · review card · banner · voice
```

## Run

**One command:** `npm run all` (or double-click `start.bat`). It starts the CLIP backend and the web app, waits until the model is loaded, reuses anything already running, and opens the app in your browser. Options: `NO_OPEN=1`, `WEB_PORT=`, `API_PORT=` (the page then needs `?api=http://127.0.0.1:PORT`, which the launcher prints). A 5-minute demo script is in [DEMO.md](DEMO.md).

Or in two terminals, as before:

Two terminals (the backend downloads CLIP once, about 600 MB, then works offline):

```bash
npm run backend      # CLIP judge on http://localhost:8000  (first start ~40 s, later ~12 s)
npm start            # web app on http://localhost:5173
```

Open http://localhost:5173 → choose your camera(s) → **Start camera(s)** (or **Video file…**). The top-right pill shows `CLIP ✓ …` when the backend is connected; if it is offline the app falls back to on-device geometry only. Click the pill to point at a different backend URL (e.g. a GPU box).

Setup (once): `python -m venv backend/.venv && backend\.venv\Scripts\pip install -r backend/requirements.txt` (use the CPU torch index: `--index-url https://download.pytorch.org/whl/cpu`). Model: `CLIP_MODEL=openai/clip-vit-large-patch14` is more accurate but ~4× slower.

Tests: `npm test` (108 Node tests: detectors, engine, rulesets, verdict handling, multi-camera logic, fighter identity, penalty audit, report, throw decisions) · `npm run test:backend` (52 tests: geometry, fusion, kicks/knees, multi-view, API, startup).

## The seven questions (backend/app/judge.py)

Each candidate strike is judged on exactly these, returned with a confidence and the evidence behind it (for kicks and knees a question 0 comes first, see below):

| # | Question | How it is answered |
|---|---|---|
| 1 | Who threw it? | pose tracker: arm extended fast towards the other fighter, re-checked by the backend |
| 2 | Intended target | the other fighter in frame (none → nothing to score) |
| 3 | Contact? | wrist-to-body-zone distance × CLIP "contact vs miss" on crops around the impact (3 frames) |
| 4 | Where? | zone softmin (head / torso / below belt / back / arms-gloves / none; facing away ⇒ back) × CLIP location prompts |
| 5 | Legal area? | head or front/side torso above the belt = legal; below belt, back, arms/gloves = not. Legs are legal only for kicks where the ruleset allows leg kicks (sent by the browser as `legal_zones`) |
| 6 | Legal glove part / striking surface? | punch: CLIP on a glove crop × geometry for elbow/forearm, knuckle part = legal · kick: foot or shin · knee: knee |
| 7 | Genuine punch? | clean vs block / miss / push / glancing: guard-glove position, wrist speed, two-hand push, punch line vs zone centre, opponent recoil × CLIP |

Fusion is a weighted geometric mean per question (`w_geo` in `backend/app/config.py`), so geometry and CLIP both have to agree for a high score. Overall: **scored** if all four of (3, 5, 6, 7) are true and their combined confidence ≥ 0.55; **review** (shown on the card with *Award / Dismiss* buttons) when borderline; a **foul** (`low_blow`, `back_head`, `open_glove`, `elbow`, `kick_below_waist`, `knee_joint_kick`) when the contact is illegal. The response also returns the crops CLIP saw (shown as thumbnails).

How a verdict is used per sport:

| Strike | Boxing | Judo | Kickboxing |
|---|---|---|---|
| Punch | clean punch = point; foul / review flag otherwise | a punch that connects = hansoku-make | same as boxing |
| **Kick** | **disqualification** | **hansoku-make** | clean kick = point (leg kick where legal; groin / knee-joint kicks and, in Full Contact, leg kicks are fouls) |
| **Knee** | **disqualification** | **hansoku-make** | clean knee = point (a foul in WAKO Full Contact) |

Contact during a break or after the bell is a foul. If the backend is offline the browser falls back to its own geometry call (a kick that lands is still a disqualification in boxing / judo).

### Several cameras (multi-angle judging)

Add up to 3 camera angles (**+ Add camera angle**, pick a camera in each row). The big view shows one camera; the others appear as small views you can click to enlarge.

- **Keeping RED and BLUE the same person in every camera.** Identity uses **clothing colour** as well as position. Each camera learns the colours of each fighter's torso while they are apart; from then on RED / BLUE follow the clothes, so the labels survive clinches and crossings, and **camera 2 and 3 take RED / BLUE from camera 1's colours**, whichever side they stand on, without being told. Camera 1 decides who RED is at the start (the fighter on its left, unless its tick box is ticked). The small coloured dots next to RED and BLUE (top-left of the video, and under each small view) show the colour each camera has learned, so you can check by eye. Wear clearly different colours (for example red vs blue, or dark vs light).
- **When clothes look alike** (both in black, say) the colour evidence is ignored and position decides, as before. Each camera's *RED on the right (only used when clothes look alike)* tick box is then the manual override. The whole feature can be switched off with *Match fighters by clothing colour*.
- **One strike, many cameras.** Each camera runs its own pose tracker. Triggers from different cameras that agree on who struck, with what, and when are merged into one strike (cameras share one clock). When any camera fires, the clips of **all** cameras around that moment are sent, so a camera that did not fire can still supply the best view of whether the strike touched.
- **Each view gets a weight** from the backend: how visible the key joints are, and how *side-on* the strike is (a limb pointing at the lens looks short and hides the gap between glove and face). Per-question probabilities are fused with a quality-weighted geometric mean.
- **A clear miss from a well-placed camera overrules an apparent hit from another angle** (a "contact veto"), because a single 2D view cannot tell a punch that lands from one that passes just in front of the face. Two good cameras that disagree therefore give "no contact".
- A camera that cannot see the opponent is ignored. Falls, throws and holdings seen by several cameras are only counted once.
- The review card lists each camera's weight, visibility, side-on score and own call, and shows every camera's screenshots (`cam2 impact` etc.).

Latency: the cost is CLIP encoding each camera's crops, about **1.3 s per camera in a quiet moment at ViT-B/16 on this CPU, and 3 s or more when other programs are busy**. A strike is judged a moment after it happens, not instantly. For 2-3 cameras consider the lighter model `CLIP_MODEL=openai/clip-vit-base-patch32` (about 4x less compute per image by design; not measured here, and less accurate). The backend also encodes every prompt set at startup so the first strike of each type is not slower.

Multi-camera works with live cameras only (they share a clock); a video file is single-view.

### Kicks and knees

- **Candidate detector (`src/detect/kick.js`)**: a foot or knee that swings fast relative to the hip, rises well above its standing height and (for a kick) ends fairly straight; a knee strike is a fast knee drive with the leg still folded. It is deliberately sensitive.
- **Question 0 (backend): is it really a kick, or just a step?** Pose kinematics (speed, leg lift) × CLIP "kick vs walking / stance" on the crops. A disqualification in boxing / judo is only issued if this passes, so walking, shuffling and bouncing do not disqualify anyone.
- **Throws are not kicks**: Osoto-gari's reaping leg is held back until the throw resolves and then dropped; the flailing legs of a fighter who is tipping over are ignored.

## Judo throw recognition (trained model)

The pose-based rule only knows one throw (Osoto-gari, from the leg reap). A model trained on the `adenhaus/judo_throws` dataset now names **ippon seoi nage, o goshi, osoto gari and uchi mata**: CLIP embeddings of 8 frames, a small temporal classifier, served at `/api/throw_type` (the backend loads `backend/models/throw_clip.pt` if it exists; `/api/health` shows `throw_model`).

- **When it is used (Judo only):** a fighter goes down firmly (torso tilt >= 55°, shoulder drop >= 1.0 torso/s) next to the opponent and the leg-reap rule has not already scored it. The browser sends the last ~2.4 s of frames; the model's answer scores a throw (graded ippon / waza-ari / yuko from the landing, named after the technique) **only if its confidence is >= 50 %**. Otherwise nothing is scored and the call log says what it saw.
- **Accuracy (honest):** 83.5 % +/- 3.6 % in grouped cross-validation, 78 % on 32 clips sent through the live endpoint. The dataset's own train/test folders overlap in source recordings and give inflated 95 %+ numbers; do not quote them. Full method, numbers and limits: [training/README.md](training/README.md).
- **Not validated on live video.** The model has no "not a throw" class and the training clips are screen recordings; a live camera will differ.

## Scorecard, replay and match report

- **Scorecard:** after each round the scoreboard shows the 10-point-must card (boxing, kickboxing) with the strikes, knockdowns and deductions behind each score, and a running total.
- **Replay:** every judged strike has **▶ Replay in slow motion**: the frames the judge looked at, all cameras side by side, the impact frame highlighted. Replays are kept in memory for this session (the newest 15 strikes); the stills and the seven answers are saved permanently.
- **📄 Match report:** opens a printable page (Print / Save as PDF): result, per-fighter summary, scorecard, penalties, the full timeline (AI vs referee calls), and every strike judged in this bout with screenshots and the seven answers. All text is escaped, and it states that the human referee's decision is final.

## Teammates' trained models (kick detector and punch type)

Two models trained by the team are part of the backend now. They load automatically when their files exist, each is skipped if missing, and `/api/health` shows `kick_model` / `punch_model`.

- **Kick detector** (`backend/models/kick_detector.joblib`, `app/kick_clf.py`): a CLIP **ViT-B/32** image embedding into a logistic regression with a tuned threshold (0.506), trained on a kick / no-kick image set. It is a **second opinion on "is this really a kick, not a step"**. For every kick candidate the 5 frames around the impact are cropped around the two fighters and scored, and the team's rolling window (`app/temporal_smoothing.py`: at least 3 of 5 frames) makes the call. Its probability is fused (weight 0.5, `KICK_MODEL_WEIGHT`) with the pose geometry and CLIP's zero-shot read, and shown in the review card ("Trained kick detector: 4 of 5 frames show a kick"). It needs ViT-B/32 while the main judge uses ViT-B/16, so the backend loads both (`app/b32_clip.py`; about 600 MB of extra weights, downloaded once by `transformers`; the original `clip` package is no longer needed). `python -m app.predict_kick <image>` still scores a single picture.
- **Punch type** (`boxing_ai/checkpoints/skeleton_gru_normalized_velocity_best.pth`, `app/punch_gru.py`): a bidirectional GRU on 25 frames of the thrower's skeleton (17 COCO keypoints, hip-centred, torso-scaled, position + velocity) that names the punch: cross, jab, lead hook, lead uppercut, rear hook or rear uppercut. It is blended 80/20 with a CLIP B/32 zero-shot read of the impact crop when that model is available. Shown as "Punch type (experimental)" in the review card and report. **It is a label only and never changes the score.** The browser now also sends the eye keypoints the GRU needs.
- **Honest accuracy:** the punch-type checkpoint's own validation macro-F1 is about **0.43** (held-out videos V1 / V4), so treat the label as a hint; the 80/20 blend was never calibrated. The datasets behind both models are **not in the repository** (only the trained files), so neither model could be measured here on real footage. They were checked for plausibility (judo frames without kicks mostly score below the kick threshold) and for correct wiring (tests with stand-in embedders plus live API calls). `boxing_ai/` is the teammate's original prototype and training scripts, left untouched; it needs the `clip` package and the BoxingVI data, which are not included.

## Finding both fighters (uploaded videos and clinches)

MediaPipe often returns ONE skeleton when two fighters are clearly in view (once it tracks one body it rarely re-detects the other, and a half-hidden fighter is dropped). The **Find 2nd fighter** option (`src/pose.js`; switched on automatically for uploaded videos, **off by default for live cameras**, see below) fixes part of that: when exactly one skeleton comes back, the first fighter is blanked out and the picture is searched again (the blanked frame, then the plain left and right parts of the frame). A found skeleton is only accepted if it passes the checks in `src/second.js`: similar body size, close to the first fighter, inside the frame, well-supported shoulders / hips, at least one knee visible, a hip line not far *above* the first fighter's unless the bodies overlap (that rejects spectators and banners but keeps a fighter lifted in a throw) and it has to repeat in the same place on two searches in a row.

- **Measured** on 1,098 sampled frames of real two-person judo footage (30 clips from the throw dataset, Python MediaPipe, same model): both fighters found in **7 %** of frames without it and **33 %** with it. With looser sanity checks the figure was about 39 %, but the extras included skeletons on logos and spectators, so the checks were tightened. The clips are mostly tangled throws, the hardest case; a clear side-on shot of two people should do better.
- **Cost and defaults:** the search costs roughly 80-100 ms a frame, so the frame rate drops from about 15 fps to about 7 fps (measured in the app; once the second fighter has been found, later searches only look in a box around him, which is cheaper). The punch detector needs about 10-12 frames a second (simulated: all 3 punches found at 12 fps, about 1 of 3 at 8 fps, none at 4 fps), so **for live Boxing / Kickboxing the search is off by default**. **Judo** tolerates a lower rate (throw found 7 of 8 times at 6 fps, falls 8 of 8 down to 4 fps) and needs both fighters, so there it is **on by default**. **Uploaded videos** always use it, and the playback speed lowers itself automatically (the HUD shows "video at 0.33×") so that about 15 frames per second of footage are analysed whatever the computer's speed. The HUD turns orange with a warning when a live camera drops under the sport's minimum (12 fps, Judo 6 fps). Ticking or unticking the box yourself overrides these defaults.
- **Limits:** it cannot help when MediaPipe merges two bodies into one skeleton, or when one fighter is almost completely hidden. A few false skeletons can still slip through (they feed the same detectors and can be undone like any AI call). Not verified on a live camera.

## Penalties: what is called automatically

`test/penalties.test.mjs` pins this list (every automatic penalty is shown to reach the scoreboard; everything else is referee-override only).

| Sport | Automatic | Referee-override only |
|---|---|---|
| Boxing (11 of 14) | low blow, hit after the bell, holding, throwing, elbow, open glove, back of head, **hitting a downed fighter**, kicking / kneeing (disqualification), **headbutt**, **ducking below the belt** | mouthguard, flagrant, unsportsmanlike |
| Kickboxing (13 of 22) | low blow, hit after the bell, holding, throwing, elbow, open glove, back of head, **hitting a downed fighter**, kick below the waist (Full Contact), illegal knee (Full Contact), thrusting kick to the knee joint, **headbutt**, **holding the opponent's leg** | throat, ropes, timidity, delay, disobeying the referee, thumbing, abuse, flagrant, unsportsmanlike |
| Judo (4 of 10) | striking, kicking / kneeing (hansoku-make), **non-combativity (30 s, shido)**, **leg grab below the belt (shido)** | false attack, stepping out, illegal grips, defensive posture, dangerous throw, disregarding the referee |

- **Hitting a downed fighter.** A contact strike thrown after the opponent has hit the mat is a foul. The strike's own peak time is compared with the moment he went down, so the punch that caused the knockdown (judged a second later) is never penalised.
- **Judo: a throw is not a kick.** Osoto-gari, Ouchi-gari, Uchi-mata and foot sweeps all swing a leg at the opponent and look like a kick to a camera. A kick or knee is only called hansoku-make in Judo if it (1) is confirmed as a real strike, (2) actually touched the opponent, (3) did not land on the legs or knee (that is a sweep / reap) and (4) the opponent did **not** go down within 3 s before or after it. The call therefore waits up to 3 s after the clip is judged; if the opponent falls in that time the log says "judged a throw attempt, not a kick". Boxing and kickboxing are unchanged (boxing: disqualification, kickboxing: points). Remaining risk: a reap that lands on the torso or hip area while the fall is not detected will still be called; use Undo.
- **Judo non-combativity.** A fighter who makes no attempt to engage (stepping in towards the opponent, or a fast leg swing at them) for 30 s gets a shido. The timer runs only while both fighters are on their feet and in view (it pauses for ground work or when nobody is tracked), restarts on any attack or throw, and a repeat after another 30 s adds the next shido. Steps are measured along the picture's horizontal axis, so it is far more reliable from a camera that sees the pair side-on than one looking straight down the line between them.
- **Pose-geometry fouls (headbutt, leg grab / hold, ducking low)** come from the skeletons alone (`src/detect/foul.js`, thresholds in `config.js` under `foul`), with no image model:
  - *Headbutt (boxing, kickboxing):* both heads meet (noses within 1 torso length) after closing fast, the attacker's head travelled at least 1.8x further than his own shoulders (a thrust, not a lunge or lean) and it was mostly his own movement (not the opponent walking onto him). Nobody on the mat.
  - *Leg grab (judo, shido):* a wrist stays within 0.2 torso lengths of the opponent's thigh or shin, below his belt line, for 0.7 s. *Leg hold (kickboxing):* the same grip kept for 2.5 s; catching a kick briefly is legal and is not called.
  - *Ducking below the belt (boxing):* the head kept below the opponent's belt line for 0.7 s while engaged and not bent flat on the mat.
  - **How far to trust them: not very.** A 2D skeleton cannot tell intent, or a deliberate butt from heads clashing in a clinch, or a grip from a hand resting near a thigh; and MediaPipe often loses one fighter when they tangle (no call is made unless both are tracked). They are checked on synthetic skeletons only (reliable across 10-30 fps and pose noise, no false calls on walking, jabs or standing still), **not on real footage**. Treat each as an AI suggestion and use Undo if it is wrong. Side-on cameras work best. Judo grips (pistol grip, bear hug) are not visible in a skeleton and stay manual.
- **↩ Undo last penalty** (scoreboard and Referee override tab): the referee can withdraw the latest penalty, which restores the score and penalty counts, and, if it ended the bout (e.g. a wrong disqualification), the bout itself. Use it for any automatic call that is wrong.

## Other detection (browser, `src/detect/`)

Kick / knee candidates, falls/knockdowns, the Osoto-gari throw (leg reap + backward fall → ippon / waza-ari / yuko) and holding. Rulesets for Boxing / Judo / Kickboxing are in `src/rules/` with the rule summaries, sources and "verify" notes shown in the Rulebook tab.

## What is and is not verified

- Verified: backend logic on synthetic skeleton clips (head / body / low / miss / block / push / glancing / no opponent), the real CLIP model loading and running on CPU (about 1 s per punch), and the full browser → backend → review card → scoreboard path with a synthetic clip.
- **Not verified: accuracy on real boxing footage.** No real fight video or webcam was available. CLIP is used zero-shot and its scores are uncalibrated; on the test images it mostly acts as a tie-breaker. CLIP is weak at glove orientation (question 6), so that answer is low-confidence evidence. Expect to tune `w_geo`, thresholds (`SCORE_THRESHOLD`, `REVIEW_THRESHOLD`, `FOUL_THRESHOLD`) and the prompts in `backend/app/prompts.py` on your own clips.
- **The new automatic penalties are verified on synthetic data only.** Judo passivity in particular is a proxy (movement towards the opponent, not grips or intent), so expect wrong calls until it has been tuned on real judo; `attack` in `src/config.js` holds its thresholds and Undo is there for the misses.
- **MediaPipe rarely sees both fighters during a throw** (both found in 2 % of frames on the throw dataset with continuous tracking). The leg-reap Osoto-gari rule needs both and will often not fire on real throws; the trained throw model above does not depend on skeletons.
- **Colour-based identity is verified on synthetic images and simulated video, not on real people.** Lighting differences between cameras, patterned or two-tone clothing, and fighters in similar colours will reduce how well it works (it then falls back to position). Only the torso is sampled.
- **Multi-camera is verified with synthetic clips and simulated camera streams, not with real cameras.** The camera rows, per-camera trackers, thumbnails and the fusion logic were run end to end (with the real CLIP model), but no real second camera or real footage was available: the quality weights, the contact veto and the camera-placement assumptions (e.g. "RED on the right") are untested on real data. Cameras are not calibrated to each other, so this is multi-view voting, not 3D triangulation.
- Only straight punches (jab/cross) trigger a punch judgement; hooks and uppercuts are not detected as candidates. **Kick / knee detection is new and untested on real footage**: its thresholds (`kick` in `src/config.js`, sliders in the Tuning tab) were set on synthetic skeletons, and exaggerated marching or high knees can still look like a knee strike to the pose tracker (the CLIP check is the safeguard). Checked kicks, push kicks (teeps) and spinning techniques are not modelled separately.
- Judo shido count is config (sources disagreed). Kickboxing: only the ABC Unified (pro) rules were read from a primary source; WAKO Full Contact / Low Kick values (round length, penalty ladder, knees in Low Kick) are partly unconfirmed: see the Rulebook tab. Use the referee-override buttons whenever the AI misses or mis-reads a strike.

## Layout

`backend/app/` CLIP judge (`judge.py`, `geometry.py`, `clip_model.py`, `prompts.py`, `main.py`) · `src/` browser detection, engine, rules, clip capture · `public/` UI + vendored MediaPipe · `test/` Node tests (`test/fixtures/skeleton.js` is the synthetic-skeleton generator used only by tests) · `backend/tests/` Python tests
