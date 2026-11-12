# TensorFold-Swarm Test Plan: Testing Session 1 (asynchronous, Fri Nov 13 to Sun Nov 15)

**Build under test:** https://aiforgames2026.github.io/DistributedMoEEngine/ (guided mission, simulated, runs in the browser). Testers need no AI background and install nothing.

**Changed Nov 11:** the session moved from a live class to asynchronous forum testing. Testers play on their own and reply to a testing post (one per dev, in both class forums), so nobody reads a script aloud or watches them play. The page now carries what the facilitator did: the mission's intro asks testers to remember where they got unsure, and the result line records what an observer would have timed.

## Testing post

**Title:** TensorFold-Swarm: guided mission (distributed AI inference simulator)

**What it is:** a browser simulation of one large AI model split across several machines, driven by speeds measured on the real hardware. You find out what slows it down and fix it.

**How to play:** open the link in a desktop browser (Chrome, Edge, or Firefox; the 3D view needs WebGL), press **Start guided mission**, and follow the card on the right. The controls are in the left panel. Goal: finish three short experiments, then get the final challenge to the target speed.

**What to focus on:** whether the mission explains itself, the moment you get unsure what to do next, and whether the 3D view shows what is going wrong.

**Length and known issues:** about 3 to 6 minutes. It works on a phone, but the 3D view, the mission card, and the controls share a small screen, so a desktop browser is better. The trace stats at the bottom left are modeled, not measured.

**Questions:** paste the result line from the last screen at the top of your reply, then answer these in order (the results screen lists them too):
1. Without scrolling back: why does data have to travel between machines?
2. Where were you least sure what to do next, and what did you do?
3. In the final challenge, what made you try what you tried?
4. In experiment 2, what changed on screen when prediction was off? From 1 to 5, how easy was it to see when an expert was stalled, and why?
5. If you ran this on your own machines, what would you want to control first: the machines, the router, the models, or the API that apps call? Why? (First press **Exit and explore freely** and try one thing for 30 seconds.)

## What each question measures

The result line looks like `TFS-alpha2 time=184s intro=21s steps=40,52,38,33 wrong=1 challenge_tries=2 hint=0 runs=6 final=292tok/s on=Chrome/Windows`: total time, seconds on the intro, seconds on each of the four steps, wrong answers, tries at the challenge, whether the challenge hint appeared, runs, final speed, and browser and OS.

| # | What we want to learn | Measured from | Good result |
|---|---|---|---|
| 1 | Does the intro explain the idea on its own? | Answer scored 0 (wrong or no idea), 1 (partly), 2 (experts live on different machines, so data goes to them); `intro=` | At least 70% score 2 |
| 2 | Can a first-timer finish unaided, and where do they get stuck? | A result line at all (finished), `time=`, the longest of `steps=`, and the step named in the answer | At least 80% finish, median under 6 min |
| 3 | Do the experiments set up the fix, or do people guess? | `challenge_tries=`, `hint=`, `wrong=`, and whether the answer cites the link or relay (reasoned) or trying menu options (guessed) | At least 60% give a reasoned answer |
| 4 | Can people read the 3D view? | Mentions red cubes yes/no; the rating and its reason | Median rating at least 4; at least 70% mention red |
| 5 | What should the dashboard do first? | Count of each choice and the reason given. Sets the order for the post-v1 dashboard (Plan.md, TASK-25) | A clear first choice, or a reason to rethink the list |

A reply without a result line counts as not finished, and the answer to question 2 says where the tester stopped.

## Recording results

After Sunday's deadline, copy one row per reply into `bench/playtest/session-1.md`, with no names. Replies come from both forums; the open forum is anonymous.

| Tester | Browser / OS | Finished? | Result line | Q1 score | Longest step | Q3 reasoned? | Q4 red? | Q4 rating | Q5 choice | Quote or note |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | | | | | | | | | | |

**Week of Nov 16:** turn each repeated problem (2 or more testers) into a Beta board issue with its evidence. Any crash or dead end on the main path goes first.

**If something breaks during the week:** a tester without WebGL sees a message saying so. Fixes redeploy through `README.md` (Deploying the website), and a fix that changes the mission bumps the result line's prefix so old and new runs stay apart.
