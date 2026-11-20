# Feedback Summary: Testing Session 1

**Build tested:** `TFS-alpha2`, the guided mission in the 3D inspector at https://aiforgames2026.github.io/DistributedMoEEngine/ ([`TestPlan.md`](TestPlan.md)).

**Sources:**
- **Testers:** nine replies to the testing post in both class forums (graded and open), Nov 13 to 15. One tester played in class and posted no result line. Testers are numbered in reply order, with no names.
- **Professor:** verbal feedback in class on Nov 13, and a forum reply on Nov 17 at 2:14 pm. Both are about the build as it was on Nov 13, when the front page was the simulated 3D inspector.

Per-tester data (result lines, step times, scores) is in [`bench/playtest/session-1.md`](bench/playtest/session-1.md).

## What changed after testing

The guided mission was built for this session, and testing showed it was the wrong focus. Most testers finished it by trying things, without learning what the engine does. Meanwhile, recorded runs of the real engine became available (TASK-16, Nov 15). So on Nov 16 the site's front page became replays of those runs (TASK-18), and the guided mission moved behind the **3D inspector** button. Its themes below are decided with that in mind: the replays and their diagram now carry the explanation, and the mission is kept as a side tool, not reworked.

## Themes

| # | Theme | Raised by | Count | Decision | Issue |
|---|---|---|---|---|---|
| 1 | [Can't tell what the diagram shows](#1-cant-tell-what-the-diagram-shows) | T1, T3, T6, professor (Nov 13) | 3 testers + professor | **Fix now** (done) | [#45](https://github.com/AIForGames2026/DistributedMoEEngine/issues/45) |
| 2 | [Are the animations accurate?](#2-are-the-animations-accurate) | professor (Nov 17) | professor | **Fix now** (done) | [#46](https://github.com/AIForGames2026/DistributedMoEEngine/issues/46) |
| 3 | [The generated reply feels off](#3-the-generated-reply-feels-off) | professor (Nov 17) | professor | **Fix now** | [#47](https://github.com/AIForGames2026/DistributedMoEEngine/issues/47) |
| 4 | [Don't know what to do in the mission](#4-dont-know-what-to-do-in-the-mission) | T1, T2, T3, T6 | 4 | **Won't fix** | [#48](https://github.com/AIForGames2026/DistributedMoEEngine/issues/48) |
| 5 | [The challenge is solved by elimination](#5-the-challenge-is-solved-by-elimination) | T2, T3, T4, T5, T8 | 5 | **Won't fix** | [#49](https://github.com/AIForGames2026/DistributedMoEEngine/issues/49) |
| 6 | [Locked controls look broken](#6-locked-controls-look-broken) | T3, T4, T8 | 3 | **Won't fix** | [#50](https://github.com/AIForGames2026/DistributedMoEEngine/issues/50) |
| 7 | [What should the dashboard control first?](#7-what-should-the-dashboard-control-first) | T4 to T8 (question 5) | 5 answers | **Fix now** (done) | [#51](https://github.com/AIForGames2026/DistributedMoEEngine/issues/51) |
| 8 | [Share the project with other developers](#8-share-the-project-with-other-developers) | professor (Nov 17) | professor | **Fix later (RC)** | [#52](https://github.com/AIForGames2026/DistributedMoEEngine/issues/52) |
| 9 | [What worked](#9-what-worked) | T1 to T9, professor | 9 + professor | **Keep** | (none) |

### 1. Can't tell what the diagram shows
- **Testers:**
  - T1: "kind of hard to tell what's going on".
  - T3: "I really just couldn't tell what was going on."
  - T6: didn't understand "how what I change has a meaningful impact".
- **Professor (verbal):** "The diagram doesn't really make much sense and doesn't tell me what's happening."
- **Decision: fix now, done.** On Nov 13 the diagram was the inspector's 3D view of a simulated cluster. Since Nov 16, the front page has a diagram of the real machines, replaying recorded runs, and it now explains itself step by step:
  - **Readable labels and links** (`d5c1fe7`, Nov 16): text drawn on a backdrop so it stays readable, stronger light-mode colors, links between machines routed around other boxes instead of crossing them, and a loading spinner.
  - **"One token" view** (`5204040`, Nov 17): follows one token through every layer, showing which experts it visits on which machine and when it crosses the cable, with a key.
  - **Prefetch and cache** (`4143dee`, Nov 18): shows experts loading from the drive, prefetched in time, or missed.

### 2. Are the animations accurate?
- **Professor:** "I am concerned that the animations aren't on point; please verify their accuracy."
- **Decision: fix now, done.** The comment is about the Nov 13 build, whose animations came from a simulator. The front page no longer animates a simulation:
  - **Real runs:** it replays telemetry recorded from the real engine on the Sparks (`d3c271c`, Nov 15; `9377196`, Nov 16). The page says when each run was recorded and that no engine is connected.
  - **Checked against the engine:** each recording's per-machine expert counts are compared with the engine's own counters.
  - **Prefetch statuses:** every step's statuses add up to the change in the cache's counters, over 1,182 and 1,128 steps (`bench/trace/2026-11-18-prefetch.md`).
  - **Still simulated:** the 3D inspector, which says so on its own panel.

### 3. The generated reply feels off
- **Professor:** "The generated content feels a bit off."
- **On the Nov 13 build:** the trace and its numbers were modeled, not generated by the engine. Since Nov 16, everything on the front page was generated by the real engine (theme 2).
- **On today's build:** the front page opens on the OLMoE-1B-7B recording. That small model's reply describes experts as models "specialized in a specific domain or task" and "trained on different subsets of data". That is a common misconception, and it contradicts the diagram beside it, where the router picks experts token by token. The larger models' recordings (Qwen3-30B-A3B, Qwen3.8-Flash-Next) answer correctly.
- **Decision: fix now.** Point out the mistake next to OLMoE's reply, and open on a recording whose reply is correct, so the first generated text a visitor reads is right.

### 4. Don't know what to do in the mission
- **Testers:**
  - T2: "I just don't know what I'm supposed to be doing."
  - T3: "felt like a weird quiz".
  - T1 and T6: unsure what was happening.
  - Intro times were short: three testers read the intro for 4 seconds or less.
  - **For balance:** T7 and T9 found it clear ("the mission card made each next step clear").
- **Decision: won't fix.** The mission is no longer the front page. The replays now explain the project (themes 1 and 2), and reworking the mission's teaching would be new content after content lock.

### 5. The challenge is solved by elimination
- **Testers:**
  - T4 changed the cluster because it "seemed to be the only option".
  - T8 first tried the greyed-out controls.
  - T3 didn't realize it was a challenge.
  - T5 was least sure there.
  - T2 needed 5 tries and the hint.
  - Only T6 and T7 reasoned from the experiments (direct TCP beats the relay).
- **Decision: won't fix**, for the same reason as theme 4.

### 6. Locked controls look broken
- **Testers:**
  - T4: "only able to change the cluster, and nothing else".
  - T8: "I assumed I was supposed to try something with the grayed out options".
  - T3: explore mode at first let them change nothing. This didn't reproduce in Chrome; it may have been the mission's locks still applying.
- **Decision: won't fix**, for the same reason as theme 4. The locks apply only inside the mission.

### 7. What should the dashboard control first?
- **Testers (question 5):** machines 2 (T6, T7), the router 1 (T4), the models 1 (T5), the API apps call 1 (T8).
- **Decision: fix now, done.** With no clear first choice, the operator dashboard (TASK-25, `dfcac16`, Nov 17) controls all four:
  - the machines (node agents)
  - the router
  - the models
  - the `/v1` API
- **Limit:** it runs from the engine on Linux, so it is not on the browser build (`bench/serve/2026-11-17-dashboard.md`).

### 8. Share the project with other developers
- **Professor:** suggested publishing on Twitter, LinkedIn or similar to get feedback from other developers.
- **Decision: fix later (RC).** Post once the RC site is final, so the link people share is the one that stays.

### 9. What worked
- **Finished:** eight testers posted a finished result line, and the ninth played in class without saving theirs.
- **Stalls read clearly:** five of seven who answered question 4 named the red cubes, with ratings of 4 and 5.
- **Polish:** T1 called the production value "crazy", and T2 and T3 called it professional and clean. The professor: "As a teaching tool, the animations are amazing."
- **Decision: keep.**

### Not themes (one tester each)
- **T8 didn't know what to give feedback on.** That's a testing-post issue; the next post will say what to look at.
- **T8's intro time (347 s) is longer than their whole run (229 s).** The intro timer probably kept counting after the mission started. Noted, not fixed, since the mission is out of focus.

## Test plan questions

| # | Question | Target | What players did or said | Met? |
|---|---|---|---|---|
| 1 | Does the intro explain the idea on its own? | 70% score 2 | 2 of 7 who answered gave the full idea (T5, T7: data goes where the experts are, or the model can't fit on one machine). T4 and T6 said only that the AI is split. T3, T8 and T9 didn't know. Median intro time 12 s. | **No** |
| 2 | Can a first-timer finish unaided? | 80% finish, median under 6 min | 8 of 9 posted a result line (89%; T9 played in class and didn't save theirs), median 3.4 min. The longest steps were experiment 3 and the challenge, and T2 and T3 finished by trying things. | **Yes**, with that caveat |
| 3 | Do the experiments set up the fix? | 60% reasoned | 2 of 7 reasoned (T6, T7), T5 in part. T4 picked the only open option, T8 tried the locked controls first, and T3 and T9 didn't know. | **No** |
| 4 | Can people read the 3D view? | Median 4+, 70% mention red | 5 of 7 named the red cubes (71%). Ratings: 4, 5 and "very easy to see". T8 described the missing prefetch in words. | **Yes** |
| 5 | What should the dashboard do first? | A clear first choice | Machines 2, router 1, models 1, API 1: no clear choice, so the dashboard covers all four (theme 7). | **No** clear choice |

## Triage

Every theme with a decision is a GitHub issue labeled `playtest`, with a priority label, an owner and a milestone, on the [board](https://github.com/orgs/AIForGames2026/projects/3); [this filter](https://github.com/AIForGames2026/DistributedMoEEngine/issues?q=label%3Aplaytest) lists them all.
- **Fix now, done:** #45, #46 and #51 (Beta), closed. Their commits were merged before the issues were opened.
- **Fix now:** #47 (Beta), open.
- **Fix later:** #52 (Release Candidate).
- **Won't fix:** #48, #49 and #50, also labeled `wontfix` and closed as not planned.
