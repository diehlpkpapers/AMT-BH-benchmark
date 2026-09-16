#ifndef __TIMERS_H__
#define __TIMERS_H__

#include "charm++.h"

#include <string.h>
#include <string>

// Per-phase wall-clock timing, spec v1.0 section 3 ("runtime of the individual
// phases"), with each phase individually disableable via -timers=.
//
// Two properties matter for an AMT comparison and shape everything below:
//
//   No added synchronization. Timing a phase by barriering around it would
//   measure the barrier, and would suppress exactly the inter-phase overlap
//   Charm++ exists to exploit. Instead every PE times its own phases locally
//   and the results are reduced once, after the last step. Nothing is
//   communicated while the simulation is running.
//
//   Disabled means absent, not zero. A masked-off phase takes no timer
//   reading, and with -timers=none the end-of-run reduction is not performed
//   either, so a timing-free run is exactly the untimed program.
//
// Because each PE's phases are consecutive intervals in its own DataManager's
// control flow, per-PE phase times do not overlap and are directly comparable
// across runtimes. They include time spent idle inside a phase, which is the
// point: that idle time is where a runtime's overlap shows up or fails to.
enum Phase {
  PHASE_INPUT = 0,      // reading and distributing the input file
  PHASE_DECOMPOSITION,  // Morton hashing, histogram/splitter refinement, exchange
  PHASE_TREEBUILD,      // octree construction and moment propagation
  PHASE_TRAVERSAL,      // tree traversal, i.e. the force interactions
  PHASE_INTEGRATION,    // KDK update, energy accumulation, per-step teardown
  PHASE_OUTPUT,         // ParaView snapshot writing
  PHASE_LOADBALANCING,  // AtSync/ResumeFromSync (not implemented yet)
  PHASE_OTHER,          // derived: the part of the step no phase claimed
  PHASE_STEP,           // whole timestep; encloses all of the above but input
  NUM_PHASES
};

// The phases that make up a step. PHASE_INPUT precedes step 0 and PHASE_STEP
// encloses these, so neither belongs in the sum.
#define FIRST_STEP_PHASE PHASE_DECOMPOSITION
#define LAST_STEP_PHASE  PHASE_LOADBALANCING

// Names as they appear both in -timers= and in the report.
inline const char *phaseName(int phase){
  static const char *names[NUM_PHASES] = {
    "input", "decomposition", "treebuild", "traversal",
    "integration", "output", "loadbalancing", "other", "step"
  };
  return names[phase];
}

// Parses a -timers= value into a bitmask: "all", "none", or a comma-separated
// list of the names above. Returns false and sets `bad` to the offending token
// if a name is not recognized.
inline bool parsePhaseMask(const std::string &spec, int &mask, std::string &bad){
  if(spec == "all"){ mask = (1 << NUM_PHASES) - 1; return true; }
  if(spec == "none"){ mask = 0; return true; }

  mask = 0;
  size_t pos = 0;
  while(true){
    size_t comma = spec.find(',', pos);
    size_t len = (comma == std::string::npos ? spec.size() : comma) - pos;
    std::string token = spec.substr(pos, len);
    if(!token.empty()){
      int phase = -1;
      for(int i = 0; i < NUM_PHASES; i++){
        if(token == phaseName(i)){ phase = i; break; }
      }
      if(phase < 0){ bad = token; return false; }
      mask |= (1 << phase);
    }
    if(comma == std::string::npos) break;
    pos = comma + 1;
  }
  return true;
}

// One of these per PE, living in that PE's DataManager.
//
// `elapsed` is a steps x NUM_PHASES table, so the report can give both
// per-step and whole-run figures without communicating per step. The input
// phase, which happens before step 0, is recorded in row 0.
class PhaseTimers {
  double *elapsed;
  double startedAt[NUM_PHASES];
  int numSteps;
  int step;
  int mask;

  public:
  PhaseTimers() : elapsed(NULL), numSteps(0), step(0), mask(0) {
    for(int i = 0; i < NUM_PHASES; i++) startedAt[i] = -1.0;
  }

  ~PhaseTimers(){ delete[] elapsed; }

  void init(int steps, int phaseMask){
    numSteps = steps;
    mask = phaseMask;
    if(mask == 0) return;
    elapsed = new double[(size_t)numSteps*NUM_PHASES];
    memset(elapsed, 0, sizeof(double)*(size_t)numSteps*NUM_PHASES);
  }

  bool enabled() const { return mask != 0; }
  bool enabled(int phase) const { return (mask & (1 << phase)) != 0; }
  int numValues() const { return numSteps*NUM_PHASES; }
  const double *values() const { return elapsed; }

  void setStep(int s){ step = s; }

  void start(int phase){
    if(!enabled(phase)) return;
    startedAt[phase] = CkWallTimer();
  }

  void stop(int phase){
    if(!enabled(phase)) return;
    if(startedAt[phase] < 0.0) return;   // never started; charge nothing
    if(step >= 0 && step < numSteps){
      elapsed[(size_t)step*NUM_PHASES + phase] += CkWallTimer() - startedAt[phase];
    }
    startedAt[phase] = -1.0;
  }

  // Ends the step: closes anything still running, then works out what part of
  // the step no phase claimed.
  //
  // Closing first means a phase whose end event a PE never reaches -- an empty
  // PE that skips part of the pipeline, say -- is charged to the end of the
  // step rather than silently reported as zero.
  //
  // The remainder is reduction waits and scheduler gaps: real time the step
  // took that no phase was running. It is computed here, per PE, rather than
  // subtracted in the report, because subtracting across-PE maxima is not
  // meaningful -- different PEs are the slowest in different phases. Note that
  // a masked-off phase's time lands here too.
  void finishStep(){
    for(int i = 0; i < NUM_PHASES; i++){
      if(i != PHASE_STEP && i != PHASE_OTHER) stop(i);
    }
    stop(PHASE_STEP);

    if(!enabled(PHASE_STEP) || !enabled(PHASE_OTHER)) return;
    if(step < 0 || step >= numSteps) return;

    double *row = elapsed + (size_t)step*NUM_PHASES;
    double claimed = 0.0;
    for(int i = FIRST_STEP_PHASE; i <= LAST_STEP_PHASE; i++) claimed += row[i];
    row[PHASE_OTHER] = row[PHASE_STEP] - claimed;
    if(row[PHASE_OTHER] < 0.0) row[PHASE_OTHER] = 0.0;
  }

  // Lays the table out for the end-of-run reduction: the same values twice,
  // so one custom reducer can produce the per-PE maximum in the first half
  // and the across-PE sum in the second.
  void fillReductionBuffer(double *buf) const {
    const int n = numValues();
    for(int i = 0; i < n; i++){
      buf[i] = elapsed[i];
      buf[n+i] = elapsed[i];
    }
  }
};

#endif // __TIMERS_H__
