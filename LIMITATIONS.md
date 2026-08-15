# Limitations & Honest Status

This document is intentionally direct. A repo that only shows favourable
numbers underserves anyone trying to build on or evaluate this project.

## By subsystem

| Area | Honest current state |
|---|---|
| **Plate detector accuracy** | Strong on its 320-image training distribution; not validated on an independent, larger, more diverse test set. 95.4% mAP@50 is a development milestone, not a deployment guarantee. |
| **Helmet & demographics models** | Trained on very small datasets (210 and 84 images respectively). Expected to generalise poorly to headgear, clothing, and lighting conditions outside the collected footage. Functional prototypes, not production classifiers. |
| **Child classification** | Relies on a scene-relative height heuristic (`geometry_utils.is_child_by_height`), not a well-validated learned category. Will misfire in low-population or unusual-camera-angle scenes. |
| **Rider/pillion assignment** | Positional (left-right) heuristic (`helmet_logic.classify_riders`). Not robust to unconventional camera angles or overtaking motorcycles. |
| **Testing rigor** | 36 unit tests cover the pure-logic components (geometry, plate validation, class voting, the two heuristics above). No integration tests against real video, no automated regression suite, no CI-driven accuracy benchmarking. |
| **Deployment readiness** | Runs reliably as a local, Docker-composable demo. Not hardened for multi-camera, 24/7, or unattended enforcement deployment — no retry/failover, no load testing performed, no rate limiting. |
| **Privacy & governance** | Demographic classification of pedestrians raises real privacy considerations for any deployment on public roads. Out of scope for this build; would need addressing (consent, data retention policy, anonymization) before any field use. |
| **Security** | Optional API-key auth exists (`IRIS_API_KEY`) but is a single shared secret, not per-user auth/RBAC. No rate limiting, no HTTPS termination configured (expected to sit behind a reverse proxy/load balancer that handles TLS). |

## Verification actually performed

- Backend and CUDA binding verified via diagnostic scripts (`verify_pipeline.py`, `verify_pipeline_debug.py`) before each run
- Pipeline run end-to-end against 18 real-world traffic video clips of varying density and lighting
- Unit tests validate the line-crossing counting math, class-vote smoothing, plate character-correction/validation, rider/pillion classification, and the child-height heuristic — all independent of the video pipeline
- Manual visual inspection of annotated output video for detection, tracking ID stability, plate reads, and helmet flags

No formal integration-test suite exists against real video inputs, and
no accuracy benchmarking runs automatically in CI (that would require
committing dataset/weights, which this repo deliberately doesn't do).

## Path to production

If you're picking this up to take further, this is the order that
matters most:

1. **Expand all three custom datasets by at least an order of
   magnitude**, sourced across multiple cities, cameras, and
   lighting/weather conditions, with a genuinely held-out test set.
2. **Replace the rider/pillion and child heuristics with learned
   classifiers** once sufficient labelled data exists.
3. **Add integration tests and load testing** on the WebSocket pipeline
   under multiple concurrent streams.
4. **Upgrade auth** from a shared API key to per-user tokens/RBAC if
   multiple operators need access, and add a data-retention/privacy
   policy for demographic and plate data before any field trial.
5. **Benchmark against an external, independently-sourced Indian
   traffic dataset** to get an unbiased accuracy estimate instead of
   validation-set numbers from the same collection process as training.

## Bottom line

The engineering — asynchronous secondary inference, multi-frame OCR
stabilisation, track-age filtering, majority-vote classification — is
sound and reflects real understanding of failure modes in real-time CV
systems. The gap between this prototype and a deployable product is
**data scale and independent validation**, not architecture.
