# Superseded by user: VGGT only

The detector fallback plan below is inactive. See `VGGT_PIPEGEN.md`.
No substitute model should be submitted as VGGT integration.

# Deadline: 15 October, 10:00 PM IST (user-confirmed)

At 8:26 PM there were 94 minutes remaining. Target: a small qualifying perception
product with real PipeGen execution. Reconstruction is an additional feature.

1. First 15 minutes: RTX laptop preflight, official PipeGen WSL2 prerequisites,
   installation and sign-in. Obtain permission for administrator changes.
   If installation is blocked, request organizer/mentor access to an already prepared
   NVIDIA workstation. This is a proposed fallback, not confirmed availability.
2. Next 35 minutes: choose curated person detection; run Requirement → PIR → Model
   → Compiler → Run on a short video. Use the default supported precision first.
   Stop adding features until the TensorRT engine and PipeGen speed report exist.
3. Next 20 minutes: emit JSON and implement an actionable presence alert within the
   generated pipeline; use a person detection to decide whether a scan should pause.
   Verify correct behavior on an empty scene and a scene with a person.
4. Next 15 minutes: integrate actual events into SPECTRA, capture the live demo,
   PIR screenshot, speed report and target environment. Keep VGGT optional if its
   model download/inference has not completed.
5. Final 9 minutes: export PipeGen project, check source repository and submission
   form; disclose incomplete features. Submit before 10 PM, not at the last second.

Suggested PipeGen requirement (review its actual generated requirements/PIR):

"Build a single-video scene-safety inspection pipeline on this NVIDIA GPU using a
curated supported person-detection model. Compile it to TensorRT and run in DeepStream.
For every processed frame, write JSON containing frame index, timestamp, original
image dimensions and detections with labels, confidence and xyxy boxes. When a person
is detected above confidence 0.5, emit a pause_scan alert; otherwise emit clear.
This alert is the primary scan-acceptance decision. Preserve generated artifacts,
TensorRT engine and PipeGen speed report. Do not infer metric distances."

Qualifying evidence is required: actual five-stage workflow, compiled engine,
speed report, GPU video run and actionable JSON. Receiving manually supplied events
or running our fixture tests is not a substitute. No claim of compliance yet.
