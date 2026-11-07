# Demo guide (about 5 minutes)

## Before you go on (2 minutes)

1. **Double-click `start.bat`** (or run `npm run all`). Wait for "AI Referee is ready". It opens the app and starts the backend.
2. The top-right pill must say **`CLIP ✓ vit-base-patch16 · cpu`**. If it says "offline", the backend is still loading (about 15 s).
3. Open the app **in Brave / Chrome / Edge, not the Claude preview** (the preview has no camera).
4. Press **🔍 Detect cameras**, allow access, pick your camera(s) in the rows, tick **RED on the right** on a camera only if clothes look alike, then **Start camera(s)**.
5. Wear **clearly different colours** (e.g. one red top, one blue/dark top): that is what keeps RED and BLUE the same person in every camera.
6. Throw one test jab at the other fighter, check a review card appears, then press **Reset** and **Clear all** so the demo starts clean.
7. Have a **recorded clip** ready (Video file…) and the latest backup zip, in case the camera misbehaves.

## The story (what to say and show)

1. **The problem.** Referees miss things; scoring is subjective; each sport has a different rulebook for the *same* action.
2. **Live punch (Boxing).** Throw a jab. Show: the ring on the video, "PUNCH LANDS", the score, and the **review card with the seven questions** (who threw it, target, contact, where, legal area, legal glove part, genuine punch) and the screenshots CLIP looked at. Press **▶ Replay in slow motion**.
3. **A foul.** Punch below the belt: **LOW BLOW → caution**, shown with the same evidence.
4. **Same kick, three rulebooks (the headline).** Throw one kick. In **Boxing** it is a **disqualification**, in **Judo** a **hansoku-make**, in **Kickboxing** it **scores a point**. Switch the preset tab and repeat (or use the review history).
5. **Judo.** An Osoto-gari is graded ippon / waza-ari / yuko from how the opponent lands. Stand still for 30 s to show **non-combativity → shido** (use the *Demo clock*; or just mention it).
6. **More than one camera.** Show the thumbnails: each strike is judged from every angle, the better-placed camera is believed, and a clear miss from a good angle overrules an apparent hit from another. RED/BLUE stay consistent by clothing colour (the small colour dots).
7. **The referee stays in charge.** Show **Referee override** (every foul and score by hand), **↩ Undo last penalty**, and **Award / Dismiss** on a borderline "needs review" card.
8. **Wrap up.** Finish a round to show the **scorecard**, then **📄 Match report** (print / save as PDF): result, scorecard, penalties, timeline and every judged strike with screenshots.

## Safe moves to demo (what is detected well)

- **Straight punches (jab, cross)**, kicks, knees, a step in towards an opponent, an Osoto-gari throw.
- **Not detected:** hooks and uppercuts (they will be ignored, not mis-scored). Keep to straight punches on stage.
- Stand roughly side-on to the camera, whole body in frame, 3 to 4 m back, good light, nothing else moving behind you.

## If something goes wrong

| Problem | Do this |
|---|---|
| Pill says "backend offline" | Wait 15 s. Otherwise close the black window and run `start.bat` again. The app still works on geometry only. |
| No cameras / only "Camera 1" | Open `/camera-test.html`; it says exactly what is blocked. Make sure you are not in the Claude preview. |
| A strike is mis-judged | Say so: it is an assistant. Use **Undo last penalty** or the override panel; that is part of the design. |
| Verdicts feel slow | Normal: about 1 to 3 s per camera on a laptop CPU. Use one camera for the live part. |
| Camera dies | Load the recorded clip with **Video file…** (use 0.5× speed if movements are fast). |

## Honest limits (say them before you are asked)

- Tuned and tested on **synthetic** data and simulated cameras. Real-world accuracy is not yet measured; thresholds are adjustable in the **Tuning** tab.
- CLIP is used **zero-shot** (uncalibrated confidences); glove-part and some fouls are low-confidence evidence.
- Detected automatically: 11 of 14 boxing penalties, 13 of 22 kickboxing, 4 of 10 judo (headbutt, leg grab / hold and ducking low are pose-geometry guesses: say so, and show Undo). The rest are referee-override.
- Colour-based identity needs clearly different clothing; otherwise it falls back to position.
