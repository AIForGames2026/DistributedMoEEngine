// Trace generator settings shared by the inspector and scripts/trace.mjs.

// Stress test for TASK-04's acceptance check: one pass over all 16 layers on
// the three-Spark preset with 64 tokens per batch, which draws about 11,000
// remote packets (dispatch plus combine).
export const STRESS = { preset: 'sparks', batch: 64, passes: 1, seed: 7 };

// The guided mission (mission.js). Fixed seed, so every tester sees the same
// numbers; scripts/trace.mjs checks that the challenge can be won only by
// changing the link. The speed multiplies how fast each run animates.
export const MISSION = {
  seed: 42, passes: 1, recall: 0.85, speed: 2, targetTps: 200,
  relayLabel: 'Today: gx10-03 + desktop, internet relay',
};
