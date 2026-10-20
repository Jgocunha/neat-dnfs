# Serial order: from-scratch runs with output feedback (19-20 October 2026)

Ten from-scratch `serial-order` runs with `--ablation output-feedback`, so mutation could wire connections out of the
output (`AblationConstants.allowOutputAsConnectionSource`, `7faae6f`). Population 1000, 200 generations,
`--target 0.93`, compatibility threshold 1.5, one run at a time, built from `face672`. Evaluator of `51d03d9` (evenly
spaced demonstration, strict one-at-a-time), before peak-height credit. Each run took 55-72 min on this machine. Data:
`data/Serial Order Output Feedback/`, 14:48 on 19 October to 01:39 on 20 October; each run's `config/README.md` has
its outcome. Champions and replays are in the hand-off bundle under `serial-order/runs/of-run-<n>/`, written by
`tools/replay/of_run_report.py`.

## Summary

- **No run solved the task.** The best re-evaluated score was 0.77; the hand-built design scores 0.994.
- **Evolution used the feedback.** In every run the best genome had a connection out of the output for 139-194 of
  200 generations. In 7 of 10 runs removing those connections from the final champion costs 0.11-0.41, and the four
  best re-evaluated runs (8, 6, 7, 3) all depend on them.
- **Three runs exploited the evaluator** with an output held just below threshold that flickers across it at the
  events (2, 9, and the output of 8). dnf-composer counts any activation above 0.00001 as a peak, and peak height was
  not scored. `f8a1098` closed this (`serial-order-peak-height-credit.md`).
- **Reported bests are inflated by noise:** up to 0.17 above the re-evaluated score (run 9, 0.832 to 0.666).
- Without a matched baseline under the same evaluator, these runs cannot show that the feedback helps, only that it is
  used. The comparison run after `f8a1098` (5 runs each arm) is meant to settle that.

## Runs

Re-evaluated: the final champion, 10 evaluations. Without feedback: the same with its connections out of the output
removed.

| Run | Reported best | Re-evaluated | Without feedback | What the output does |
|---|---|---|---|---|
| 1 | 0.613 | 0.593 | 0.598 | Faint burst: all four 34 after X, height 0.4-1.5 |
| 2 | 0.804 | 0.641 | 0.245 | Held 0.1 below threshold everywhere, flickering to 0.0-0.2 at the events |
| 3 | 0.728 | 0.673 | 0.519 | D (2.4) then B (1.3), one at a time, every trigger; A and C missing |
| 4 | 0.619 | 0.599 | 0.584 | Faint burst, D about 10 ahead |
| 5 | 0.553 | 0.390 | 0.354 | One or two tall peaks per trigger at varying events, also during the demonstration |
| 6 | 0.830 | 0.739 | 0.454 | Two faint pairs, D and A at 50-70 after X, then C and B at 120-190 |
| 7 | 0.765 | 0.679 | 0.565 | Two groups with partly right order; which events go in which group is noise-driven |
| 8 | 0.797 | 0.767 | 0.454 | Output flickers (0.0-0.3) in a noise-driven order; hidden nf 3 makes real peaks one at a time |
| 9 | 0.832 | 0.666 | 0.254 | Threshold flicker, single steps, one event at a time, 400-1200 after X |
| 10 | 0.644 | 0.603 | 0.437 | Faint burst, driven by a hidden field the output feeds |

What the feedback does when it matters: it closes a loop through hidden fields that holds the output at threshold
(runs 2, 9), carries the order (3, 7) or the grouping (6), or drives a hidden field that does the selecting (8).
Runs 1 and 4 had output-sourced connections into fields that never reach threshold.

Under the peak-height credit of `f8a1098`, runs 1, 2, 3 and 6 rescore 0.436, 0.338, 0.494 and 0.413.

Species ranged from 1 to 667 at the end; runs 3 and 8 collapsed to one species.
