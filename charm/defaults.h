#ifndef __DEFAULTS_H__
#define __DEFAULTS_H__

#define DEFAULT_THETA 0.5 
#define DEFAULT_DT 0
#define DEFAULT_DTIME 0.025
#define DEFAULT_EPS 0.05
#define DEFAULT_TOL 1.0
// G = 1 reproduces the Plummer/SPLASH unit system (M = -4E = G = 1) that
// the bundled generator emits. Datasets in physical units need the real G
// in whatever units they use, passed with -G=<value>.
#define DEFAULT_G 1.0


// Particles per TreePiece. This also sets the default TreePiece count, as
// 2*numParticles/ppc, so it decides how finely the work is cut up.
//
// It was 1000, which on anything but a very large input left the histogram
// refinement stopping after a handful of non-empty TreePieces -- all of them
// at low indices, hence all on the first PE. On 10k particles across 2 PEs
// that put all 1276 buckets on PE 0 and left PE 1 idle. 100 keeps enough
// TreePieces in play for the decomposition to spread them, and costs nothing:
// step time is flat from ppc 50 to 800 at both 10k and 100k particles.
#define DEFAULT_PPC 100
#define DEFAULT_PPB 10
#define DEFAULT_KILLAT 10
#define DEFAULT_CHUNK_DEPTH 3
#define DEFAULT_YIELD_PERIOD 5
#define DEFAULT_TREE_PIECES_PER_PROC 8

// ParaView snapshots are off unless -output= names a prefix: a full-frequency
// run writes about 100 bytes per particle per step, which is not something a
// benchmark should do by default. Once enabled, every step is written.
#define DEFAULT_OUTPUT_FREQ 1

// Every phase timed by default. -timers=none turns the instrumentation off
// entirely, including the end-of-run reduction that reports it.
#define DEFAULT_TIMERS "all"
#endif
