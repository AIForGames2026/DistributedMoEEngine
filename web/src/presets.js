// Trace generator settings shared by the inspector and scripts/trace.mjs.

// Stress test for TASK-04's acceptance check: one pass over all 16 layers on
// the three-Spark preset with 64 tokens per batch, which draws about 11,000
// remote packets (dispatch plus combine).
export const STRESS = { preset: 'sparks', batch: 64, passes: 1, seed: 7 };
