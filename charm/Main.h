#ifndef __MAIN_H__
#define __MAIN_H__

#include "Parameters.h"
#include "defines.h"
#include "OrientedBox.h"
#include "Timers.h"

#include <vector>

#include "barnes.decl.h"

class Main : public CBase_Main {
  Parameters params;
  int numQuiescenceRecvd;

  // The final energies and the phase timings arrive as two independent
  // reductions, in either order. Both are held until both have landed, so the
  // end-of-run output reads the same every time.
  BoundingBox finalEnergy;
  bool haveFinalEnergy;
  std::vector<double> phaseMax;    // slowest PE, per step and phase
  std::vector<double> phaseMean;   // averaged over PEs
  bool havePhaseTimers;

  void getNumParticles();
  void scanBinaryInput();
  void scanCsvInput();
  void setParameters(CkArgMsg *m);
  void usage();
  void finishReports();
  void printPhaseReport();

  public:
  Main(CkArgMsg *msg);
  void commence();
  void niceExit();
  void reportFinalEnergy(CkReductionMsg *msg);
  void reportPhaseTimers(CkReductionMsg *msg);
  void reportBalance(CkReductionMsg *msg);

  void quiescence();
  void quiescenceExit();
};

#endif
