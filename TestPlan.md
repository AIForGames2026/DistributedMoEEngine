# TensorFold-Swarm Test Plan: Testing Session 1 (week of Nov 9)

**Build under test:** https://aiforgames2026.github.io/DistributedMoEEngine/ (guided mission, simulated, runs in the browser). Testers need no AI background and install nothing.

## Script

**Before they start, say only this:** "This is a simulator for a system that runs one huge AI model across several machines. Open the link and follow what's on screen. It takes about 3 minutes. Think out loud if you can. I can't help you, because if something needs explaining, I want to find that out."

**Tasks for the tester**
1. Open the link in any desktop browser, in a fresh tab.
2. Complete the mission: three short experiments, then the final challenge.
3. On the results screen, copy the result line into your notes or the feedback form.
4. Press "Exit and explore freely", then try one thing you're curious about for 30 seconds.
5. Answer the post-play questions below.

**While they play:** stay silent, even if they get stuck for 60 seconds. After that, write down "needed help" and where, then give the smallest hint you can. Every explanation is a finding, not a failure.

## Questions and what to measure

| # | What we want to learn | Ask the tester after play | Observe or measure | Good result |
|---|---|---|---|---|
| 1 | Does the intro explain the idea without me? | "Without looking: why does data have to travel between machines?" | Score 0 (wrong or no idea), 1 (partly), 2 (experts live on different machines, so data goes to them). Also seconds spent on the intro. | At least 70% score 2 |
| 2 | Can a first-timer finish unaided, and where do they get stuck? | "Where were you least sure what to do next?" | Finished yes/no, `time=`, step and control where they paused over 15 s, times they needed help. | At least 80% finish with no help, median under 6 min |
| 3 | Do the experiments set up the fix, or do people guess? | "In the final challenge, what made you try what you tried?" | `challenge_tries=`, `wrong=`, and whether their reason cites the link or relay (reasoned) or trying menu options (guessed). | At least 60% give a reasoned answer |
| 4 | Can people read the 3D view? | "In experiment 2, what changed on screen when prediction was off?" and "Rate 1 to 5: how easy was it to see when an expert was stalled?" | Mentions red cubes yes/no; the rating; whether they look at the 3D view or only the numbers. | Median rating at least 4; at least 70% mention red |
| 5 | What should the dashboard do first? | "If you ran this on your own machines, what would you want to control first: machines, the router, models, or the API that apps call? Why?" | Count of each choice and the reason given. Sets the order for the post-v1 dashboard (Plan.md, TASK-25). | A clear first choice, or a reason to rethink the list |

## Recording results

The tester pastes the result line and their answers into the class feedback form. I keep one row per tester in the table below, copied into `bench/playtest/session-1.md` after the session, with observations written live and no names.

| Tester | Browser / OS | Finished? | Result line | Q1 score | Stuck at | Help needed | Q3 reasoned? | Q4 red? | Q4 rating | Q5 choice | Quote or note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | | | | | | | | | | | |

**After the session:** turn each repeated problem (2 or more testers) into a Beta board issue with its evidence. Any crash or dead end on the main path goes first.

**If something breaks in class:** if WebGL is missing, the page says so; try another browser and record the browser and OS. Any fix redeploys through `README.md` (Deploying the inspector) and goes in the release notes.
