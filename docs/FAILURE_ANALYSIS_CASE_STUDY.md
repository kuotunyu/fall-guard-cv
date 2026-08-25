# Cross-dataset generalization failure

## Question

Does the event detector retain useful fall sensitivity and non-fall
specificity when thresholds selected on URFD are transferred unchanged to the
unseen Le2i sites? This case study treats a poor transfer result as engineering
evidence. It asks what the frozen protocol can support, not how to tune the
test set until its numbers look better.

## Frozen protocol

Thresholds and timing parameters were selected using the 70 URFD videos only.
Le2i was a pure event-level test of 130 videos: 127 falls and **only 3 non-fall ADL videos**, with 97.8 seconds of total ADL footage. Subjects, sites, and
cameras do not overlap between the datasets. The transferred parameters were
`v_y=3.0`, `theta=45.0 degrees`, `falling_timeout_s=1.5`, and
`confirm_seconds=0.3`; Le2i did not participate in their selection.

The **0.3-second evaluation** confirmation setting also differs from the
**10-second runtime** default. The reported research protocol therefore does
not measure the behavior of the default runtime configuration.

## Observed failure

The frozen Le2i event-level result was **Sensitivity 0.559** with a recorded
95% interval of `[0.47, 0.64]`, and **Specificity 0.000** with a recorded 95%
interval of `[0.00, 0.56]`. All three Le2i ADL videos were false positives.
Because the negative set contains only three videos, the specificity estimate
is unstable and must be read with its wide interval. The result exposes a
cross-site generalization gap; it is **not deployment evidence** and does not
establish robustness for home or clinical use.

## Root-cause hypotheses

The frozen artifacts do not establish a causal root cause. They support the
following hypotheses for future controlled tests:

- Dataset shift may matter because subjects, scenes, and cameras differ across
  URFD and Le2i.
- Event-label timing or semantics may differ; the existing analysis explicitly
  does not claim that Le2i frame labels are equivalent to URFD labels.
- Timing transfer may matter because the URFD search selected the edge values
  `1.5` seconds for falling timeout and `0.3` seconds for confirmation.
- The three-video negative sample makes specificity highly sensitive to each
  outcome, even though it does not by itself explain why those videos fired.

These are testable explanations, not findings of causation.

## What was deliberately not tuned

No Le2i parameter tuning was performed to rescue the result. The project did
not change velocity or angle thresholds, FSM timing, labels, exclusion rules,
or test composition after seeing the Le2i outcome. This closeout also does not
download data, train a model, rerun a paid VLM comparison, or replace the
recorded metrics. Keeping the failed transfer frozen prevents test-set feedback
from being disguised as generalization.

## Engineering lessons

- Internal LOSO and cross-dataset transfer answer different questions and
  should not be reduced to a single improvement or drop score.
- A specificity claim needs enough representative negative videos; three ADL
  videos cannot support a stable deployment estimate.
- Evaluation and runtime timing must share one declared contract before a
  benchmark can support runtime expectations.
- Dataset, subject, site, and camera boundaries belong in the release evidence,
  not only in an experiment notebook.
- An event detector intended for safety-sensitive use needs explicit failure
  behavior and broader validation; a promising internal fold is insufficient.

## Next experiment contract

The next experiment remains design-only until data access and review are
separately approved. It must add subject and site diversity with substantially
more non-fall behavior, define consistent event timing across datasets, and
lock validation and test splits before parameter selection. The protocol must
predeclare a deployment threshold and its required sensitivity/specificity
evidence before the test set is opened. No tuning may use the locked test split,
and the runtime confirmation setting must match the evaluated setting or be
reported as a separate condition.
