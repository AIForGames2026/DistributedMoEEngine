# Testing Session 1: guided mission (asynchronous, Nov 13 to 15)

Six forum replies to the testing post ([`TestPlan.md`](../../TestPlan.md)), all on desktop Windows (four Chrome, one Edge, one Firefox), build `TFS-alpha2`. No names; testers are numbered in reply order. Answers are paraphrased, with short quotes.

## Replies

| Tester | Browser | Time | Intro | Steps (s) | Wrong | Challenge tries | Hint | Q1 score | Q3 reasoned? | Q4 red? | Q4 rating | Q5 choice | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | Chrome | 223 s | 4 s | 22, 51, 100, 51 | 2 | 2 | no | (skipped) | (skipped) | (skipped) | | | Declined the questions. "Kind of hard to tell what's going on," but got the gist; praised the production value. |
| T2 | Chrome | 197 s | 1 s | 14, 20, 20, 142 | 1 | 5 | yes | (skipped) | (skipped) | (skipped) | | | "I just don't know what I'm supposed to be doing"; didn't know how to go on after the tutorial. |
| T3 | Edge | 210 s | 1 s | 49, 18, 124, 20 | 0 | 1 | no | 0 | no: "I just clicked on the things" | no | | | Guessed throughout; "felt like a weird quiz" and "a test I never prepared for". Reported that explore mode first let them change nothing. |
| T4 | Firefox | 179 s | 28 s | 63, 34, 43, 38 | 0 | 1 | no | 1 | no: the cluster "seemed to be the only option" | yes | 5 (no number given; "very easy to see") | router (predictor recall) | In the challenge, only the cluster could be changed. |
| T5 | Chrome | 141 s | 16 s | 30, 34, 45, 32 | 0 | 1 | no | 2 | partly: curious what a direct connection would do | yes | (none) | models | Least sure at the challenge. |
| T6 | Chrome | 125 s | 8 s | 20, 38, 41, 26 | 0 | 1 | no | 1 | yes: direct TCP beats the internet relay | yes | 4 | machines | "I don't really understand what is going on or how what I change has a meaningful impact." |

## Against the test plan's targets

| # | Question | Target | Result | Met? |
|---|---|---|---|---|
| 1 | Does the intro explain the idea on its own? | 70% score 2 | 1 of 4 who answered (25%). Median time on the intro: 6 s, and three testers spent 4 s or less | **no** |
| 2 | Can a first-timer finish unaided? | 80% finish, median under 6 min | 6 of 6 finished, median 3.1 min. But T2 and T3 said they didn't know what to do, and finished by trying things. The longest steps: experiment 3 (100 s and 124 s) and the challenge (142 s) | met, with a caveat |
| 3 | Do the experiments set up the fix? | 60% reasoned | 1 of 4 who answered reasoned (T6), plus T5 in part. T4 picked the cluster because nothing else could be changed | **no** |
| 4 | Can people read the 3D view? | Median rating 4 or more; 70% mention red | 3 of 4 who answered mention red cubes; ratings 4 and "very easy" | met |
| 5 | What should the dashboard do first? | A clear first choice | machines 1, models 1, router 1; no clear choice | **no** |

Two of six declined the questions entirely, and a third answered "I don't know" to most of them.

## Repeated problems (2 or more testers)

1. **People don't know what they are doing or why** (T1, T2, T3, T6). The intro is read for seconds or skipped, so the experiments start before the idea lands, and the controls (Lookahead prefetch, Cluster, top-k) are named in the system's terms.
2. **The challenge is solved by elimination, not reasoning** (T3, T4, T5). During the mission only Cluster and Lookahead can be changed, and only two Cluster options are valid, so the answer is close to forced. It also reads as a quiz rather than a problem to solve (T2, T3).
3. **Locked controls look broken** (T3, T4). The mission disables trace, top-k, and recall with no explanation. T3 also reported that explore mode first changed nothing; that did not reproduce in Chrome after exiting from the intro or from the results screen, so it may have been the mission's locks or a recall slider disabled because prediction was off.

What worked: everyone finished, the red cubes read clearly (T4, T5, T6), and three testers praised how polished it looks (T1, T2, T3).
