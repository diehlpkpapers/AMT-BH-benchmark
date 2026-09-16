#include "Main.h"
#include "Parameters.h"

#include "TreePiece.h"
#include "DataManager.h"
#include "Reduction.h"
#include "Messages.h"

#include "defaults.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <limits.h>
#include <string.h>
using namespace std;

CProxy_TreePiece treePieceProxy;
CProxy_DataManager dataManagerProxy;
CProxy_Main mainProxy;

Parameters globalParams;

Main::Main(CkArgMsg *msg){
  setParameters(msg);

  dataManagerProxy = CProxy_DataManager::ckNew();

  CkArrayOptions opts(globalParams.numTreePieces);
  treePieceProxy = CProxy_TreePiece::ckNew(opts);

  mainProxy = thisProxy;

  thisProxy.commence();

  CkCallback cb(CkIndex_Main::quiescence(),thisProxy);
  CkStartQD(cb);

  numQuiescenceRecvd = 0;
  haveFinalEnergy = false;
  havePhaseTimers = false;
  delete msg;
}

static bool hasSuffix(const string &s, const string &suffix){
  if(s.size() < suffix.size()) return false;
  return s.compare(s.size()-suffix.size(), suffix.size(), suffix) == 0;
}

void Main::setParameters(CkArgMsg *m){
  map<string,string> table;
  params.extractParameters(m->argc, m->argv, table); 

  map<string,string>::iterator it = table.find("in");
  if(it == table.end()){
    CkPrintf("[Main] Must specify input particle file:\n");
    usage();
    CkAbort("bad command line arguments\n");
  }

  globalParams.filename = params.getsparam("in", table);

  it = table.find("format");
  if(it == table.end()){
    // No -format given: infer from the extension. A .csv file is the
    // whitepaper's shared text format; anything else is the bundled binary.
    globalParams.inputFormat = hasSuffix(globalParams.filename, ".csv")
                                   ? INPUT_CSV : INPUT_BINARY;
  }
  else if(it->second == "csv"){
    globalParams.inputFormat = INPUT_CSV;
  }
  else if(it->second == "binary" || it->second == "bin"){
    globalParams.inputFormat = INPUT_BINARY;
  }
  else{
    CkPrintf("[Main] unrecognized -format=%s (expected 'csv' or 'binary')\n",
             it->second.c_str());
    usage();
    CkAbort("bad command line arguments\n");
  }
  CkPrintf("format: %s\n",
           globalParams.inputFormat == INPUT_CSV ? "csv" : "binary");

  globalParams.theta = params.getrparam("theta", DEFAULT_THETA, table);
  CkPrintf("theta: %f\n", globalParams.theta);

  globalParams.G = params.getrparam("G", DEFAULT_G, table);
  CkPrintf("G: %g\n", globalParams.G);

  globalParams.dtime = params.getrparam("dtime", DEFAULT_DTIME, table);
  CkPrintf("dtime: %f\n", globalParams.dtime);

  globalParams.dthf = globalParams.dtime/2.0;

  globalParams.epssq = params.getrparam("eps", DEFAULT_EPS, table);
  CkPrintf("eps: %f\n", globalParams.epssq);
  globalParams.epssq = globalParams.epssq*globalParams.epssq;

  globalParams.tolsq = globalParams.theta;
  CkPrintf("tol: %f\n", globalParams.tolsq);
  globalParams.tolsq = globalParams.tolsq*globalParams.tolsq;

  globalParams.ppc = params.getiparam("ppc", DEFAULT_PPC, table); 
  CkPrintf("ppc: %d\n", globalParams.ppc);

  globalParams.ppb = params.getiparam("b", DEFAULT_PPB, table);
  CkPrintf("bucketSize: %d\n", globalParams.ppb);

  globalParams.iterations = params.getiparam("killat", DEFAULT_KILLAT, table);
  CkPrintf("killat: %d\n", globalParams.iterations);
  
  globalParams.cacheLineSize = params.getiparam("chunkdepth", DEFAULT_CHUNK_DEPTH, table);
  CkPrintf("chunkdepth: %d\n", globalParams.cacheLineSize);

  globalParams.yieldPeriod = params.getiparam("yield", DEFAULT_YIELD_PERIOD, table);
  CkPrintf("yieldPeriod: %d\n", globalParams.yieldPeriod);

  it = table.find("timers");
  string timerSpec = (it == table.end()) ? string(DEFAULT_TIMERS) : it->second;
  string badPhase;
  if(!parsePhaseMask(timerSpec, globalParams.timerMask, badPhase)){
    CkPrintf("[Main] -timers=: unknown phase '%s'. Expected 'all', 'none', or a "
             "comma-separated list of:\n[Main]  ", badPhase.c_str());
    for(int i = 0; i < NUM_PHASES; i++) CkPrintf(" %s", phaseName(i));
    CkPrintf("\n");
    CkAbort("bad command line arguments\n");
  }
  CkPrintf("timers: %s\n", timerSpec.c_str());

  globalParams.output.prefix = params.getsparam("output", table);
  globalParams.output.frequency =
      params.getiparam("outputfreq", DEFAULT_OUTPUT_FREQ, table);
  if(globalParams.output.enabled() && globalParams.output.frequency < 1){
    CkPrintf("[Main] -outputfreq=%d: must be at least 1\n",
             globalParams.output.frequency);
    CkAbort("bad command line arguments\n");
  }
  if(globalParams.output.enabled()){
    CkPrintf("output: %s.pvd every %d step%s\n",
             globalParams.output.prefix.c_str(),
             globalParams.output.frequency,
             globalParams.output.frequency == 1 ? "" : "s");
  }
  else{
    CkPrintf("output: none\n");
  }

  getNumParticles();

  it = table.find("p");
  if(it == table.end()){
    // A budget rather than a prediction. The splitters cut the Morton key
    // space at midpoints, not at medians, so how many leaves the refinement
    // needs depends on how clustered the input is. Measured on the bundled
    // generator, the requirement runs from about 1.3x numParticles/ppc when
    // there are thousands of bins up to about 3.2x when there are only a
    // handful -- with few bins there is no averaging out. The additive term is
    // what covers that small end; the multiplier alone never does.
    //
    // Overshooting costs empty TreePieces. Undershooting no longer aborts: the
    // decomposition just comes out coarser, with a warning.
    globalParams.numTreePieces =
        2*(globalParams.numParticles/globalParams.ppc) + 16;
  }
  else{
    globalParams.numTreePieces = atoi(it->second.c_str());
  }

  // More TreePieces than bodies cannot all be filled, and the leftovers are
  // not harmlessly spread around: Charm++'s default map gives each PE a
  // contiguous block of array indices, so an unfillable top of the array is an
  // idle top of the machine. Asking for 4096 TreePieces for 2000 bodies on 4
  // PEs left the last PE with nothing at all.
  if(globalParams.numTreePieces > globalParams.numParticles){
    CkPrintf("[Main] %d TreePieces asked for but only %d bodies to fill them; "
             "using %d\n",
             globalParams.numTreePieces, globalParams.numParticles,
             globalParams.numParticles);
    globalParams.numTreePieces = globalParams.numParticles;
  }

  // Fewer TreePieces than PEs guarantees idle PEs whatever the decomposition
  // does. This has to come last: it is the one case where an unfillable
  // TreePiece is still better than a PE with no TreePiece to be given.
  if(globalParams.numTreePieces < CkNumPes())
    globalParams.numTreePieces = CkNumPes();

  CkPrintf("tree pieces: %d\n", globalParams.numTreePieces);


}

void Main::getNumParticles(){
  if(globalParams.inputFormat == INPUT_CSV) scanCsvInput();
  else scanBinaryInput();
}

// Reads the CSV header (if there is one) to learn which column holds what,
// then counts records. Both are one-time, PE 0, I/O-bound costs that land
// before the simulation proper -- the spec excludes input time from the
// reported metrics.
void Main::scanCsvInput(){
  CkPrintf("[Main] file %s\n", globalParams.filename.c_str());
  ifstream partFile(globalParams.filename.c_str(), ios::in | ios::binary);
  CkAssert(partFile.is_open());

  CsvLayout &csv = globalParams.csv;

  string firstLine;
  if(!std::getline(partFile, firstLine)){
    CkAbort("input file is empty\n");
  }

  if(CsvLayout::looksLikeHeader(firstLine)){
    string missing;
    if(!csv.parseHeader(firstLine, missing)){
      CkPrintf("[Main] %s: header names no '%s' column. Expected the columns "
               "of spec v1.0 sec. 5: id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z\n",
               globalParams.filename.c_str(), missing.c_str());
      CkAbort("unusable CSV header\n");
    }
    csv.dataOffset = (long)partFile.tellg();
  }
  else{
    // No header. Take the spec's column order on faith, but say so, because
    // a misordered headerless file would otherwise run and give wrong answers.
    csv.setSpecOrder();
    csv.dataOffset = 0;
    CkPrintf("[Main] no CSV header found; assuming spec column order "
             "id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z\n");
  }

  partFile.clear();
  partFile.seekg(0, ios::end);
  globalParams.inputFileSize = (long)partFile.tellg();
  partFile.seekg(csv.dataOffset, ios::beg);

  // Count records by scanning for line terminators rather than by parsing --
  // this pass only needs to know how many there are. Blank lines do not
  // count, matching isBlankCsvLine() in the per-PE readers.
  const size_t BUFSIZE = 1 << 20;
  vector<char> buf(BUFSIZE);
  long long records = 0;
  bool sawContent = false;
  while(partFile){
    partFile.read(&buf[0], BUFSIZE);
    std::streamsize got = partFile.gcount();
    for(std::streamsize i = 0; i < got; i++){
      char c = buf[i];
      if(c == '\n'){
        if(sawContent) records++;
        sawContent = false;
      }
      else if(c != '\r'){
        sawContent = true;
      }
    }
  }
  if(sawContent) records++;   // last line, unterminated

  partFile.close();

  if(records == 0){
    CkAbort("input file contains no particle records\n");
  }
  if(records > (long long)INT_MAX){
    CkAbort("more particles than an int can index\n");
  }
  globalParams.numParticles = (int)records;
  CkPrintf("[Main] numParticles %d\n", globalParams.numParticles);
}

void Main::scanBinaryInput(){
  ifstream partFile;
  CkPrintf("[Main] file %s\n", globalParams.filename.c_str());
  partFile.open(globalParams.filename.c_str(), ios::in | ios::binary);
  CkAssert(partFile.is_open());

  partFile.read((char *)(&globalParams.numParticles),sizeof(int)); 
  CkPrintf("[Main] numParticles %d\n", globalParams.numParticles);

  // The on-disk width of this format follows sizeof(Real), so a file written
  // by an older single-precision build would be silently misread. Check the
  // size before anything seeks into it.
  partFile.seekg(0, ios::end);
  std::streamoff actualSize = partFile.tellg();
  std::streamoff expectedSize =
      (std::streamoff)PREAMBLE_SIZE +
      (std::streamoff)globalParams.numParticles*(std::streamoff)SIZE_PER_PARTICLE;
  if(actualSize != expectedSize){
    CkPrintf("[Main] %s: expected %lld bytes for %d particles at %zu bytes each, found %lld.\n",
             globalParams.filename.c_str(), (long long)expectedSize,
             globalParams.numParticles, (size_t)SIZE_PER_PARTICLE,
             (long long)actualSize);
    CkPrintf("[Main] This build reads/writes %zu-byte reals. Regenerate the input "
             "with the matching ./plummer or ./gen.\n", sizeof(Real));
    CkAbort("input file size does not match this build's particle layout\n");
  }

  partFile.close();
}

void Main::commence(){
  CkPrintf("[Main] load particles\n");
  CkReductionMsg *redMsg;
  dataManagerProxy.loadParticles(CkCallbackResumeThread((void *&)redMsg));

  BoundingBox &universe = *((BoundingBox *)redMsg->getData());
  Real pad = 0.001;
  universe.expand(pad);
  CkPrintf("[Main] universe bb: %f %f %f %f %f %f\n",
                                universe.box.lesser_corner.x,
                                universe.box.lesser_corner.y,
                                universe.box.lesser_corner.z,
                                universe.box.greater_corner.x,
                                universe.box.greater_corner.y,
                                universe.box.greater_corner.z
                                );

  dataManagerProxy.decompose(universe);

}

void Main::reportFinalEnergy(CkReductionMsg *msg){
  finalEnergy = *((BoundingBox *)msg->getData());
  delete msg;
  haveFinalEnergy = true;
  finishReports();
}

void Main::reportPhaseTimers(CkReductionMsg *msg){
  const int n = (int)(msg->getSize()/sizeof(double))/2;
  const double *data = (const double *)msg->getData();
  const double npes = (double)CkNumPes();

  phaseMax.assign(data, data+n);
  phaseMean.resize(n);
  for(int i = 0; i < n; i++) phaseMean[i] = data[n+i]/npes;

  delete msg;
  havePhaseTimers = true;
  finishReports();
}

// Particles per PE after the first step's decomposition. Printed as it lands,
// not held for finishReports(), because it describes the start of the run and
// is worth seeing while the run is still going.
void Main::reportBalance(CkReductionMsg *msg){
  const int *counts = (const int *)msg->getData();
  const int npes = CkNumPes();

  int total = 0, empty = 0;
  int lo = counts[0], hi = counts[0];
  for(int i = 0; i < npes; i++){
    total += counts[i];
    if(counts[i] < lo) lo = counts[i];
    if(counts[i] > hi) hi = counts[i];
    if(counts[i] == 0) empty++;
  }
  const double mean = (double)total/npes;

  const int ntp = counts[npes];
  const double tpMean = ntp > 0 ? (double)total/ntp : 0.0;

  CkPrintf("[BALANCE] step 0: %d particles over %d PEs in %d of %d TreePieces; "
           "per PE min %d max %d mean %.0f, max/mean %.3f; "
           "per TreePiece min %d max %d mean %.1f, max/mean %.3f",
           total, npes, ntp, globalParams.numTreePieces,
           lo, hi, mean, mean > 0.0 ? hi/mean : 0.0,
           counts[npes+1], counts[npes+2], tpMean,
           tpMean > 0.0 ? counts[npes+2]/tpMean : 0.0);
  if(empty > 0) CkPrintf(", %d PE%s empty", empty, empty == 1 ? "" : "s");
  CkPrintf("\n");

  delete msg;
}

// The energy reduction and the timer reduction race each other; whichever
// lands second does the printing, so the output order is fixed.
void Main::finishReports(){
  if(!haveFinalEnergy) return;
  if(globalParams.timerMask != 0 && !havePhaseTimers) return;

  CkPrintf("[ENERGY] step %d E_K %.10g E_P %.10g E_T %.10g\n",
           globalParams.iterations-1,
           finalEnergy.kineticEnergy,
           finalEnergy.potentialEnergy,
           finalEnergy.totalEnergy());

  if(globalParams.timerMask != 0) printPhaseReport();

  niceExit();
}

void Main::printPhaseReport(){
  const int steps = globalParams.iterations;
  const int mask = globalParams.timerMask;

  CkPrintf("[TIMERS] wall clock by phase over %d step%s on %d PE%s, seconds.\n",
           steps, steps == 1 ? "" : "s",
           CkNumPes(), CkNumPes() == 1 ? "" : "s");
  CkPrintf("[TIMERS] 'max' is the slowest PE in that phase, which is what sets "
           "the critical path;\n");
  CkPrintf("[TIMERS] 'mean' averages over PEs, so max-mean is the imbalance. No "
           "barriers were\n");
  CkPrintf("[TIMERS] added: each PE timed itself and the tables were reduced "
           "once, after the run.\n");

  // Per-step lines first: the spec's "runtime of one timestep" metric, one
  // machine-readable line each, listing only the phases that were enabled.
  for(int s = 0; s < steps; s++){
    const double *row = &phaseMax[(size_t)s*NUM_PHASES];
    CkPrintf("[STEP] step %d", s);
    for(int ph = 0; ph < NUM_PHASES; ph++){
      if(!(mask & (1 << ph))) continue;
      if(ph == PHASE_INPUT && s != 0) continue;   // input happens once
      CkPrintf(" %s %.6f", phaseName(ph), row[ph]);
    }
    CkPrintf("\n");
  }

  CkPrintf("[TIMERS] %-14s %12s %12s %12s %8s\n",
           "phase", "total(max)", "per step", "total(mean)", "share");
  for(int ph = 0; ph < NUM_PHASES; ph++){
    if(!(mask & (1 << ph))) continue;

    double totalMax = 0.0, totalMean = 0.0;
    for(int s = 0; s < steps; s++){
      totalMax += phaseMax[(size_t)s*NUM_PHASES + ph];
      totalMean += phaseMean[(size_t)s*NUM_PHASES + ph];
    }

    // Shares are of the step total, so input -- which precedes step 0 -- has
    // none, and neither does step itself.
    double stepTotal = 0.0;
    if(mask & (1 << PHASE_STEP)){
      for(int s = 0; s < steps; s++) stepTotal += phaseMax[(size_t)s*NUM_PHASES + PHASE_STEP];
    }

    char share[16];
    if(ph == PHASE_INPUT || ph == PHASE_STEP || stepTotal <= 0.0) strcpy(share, "-");
    else snprintf(share, sizeof(share), "%.1f%%", 100.0*totalMax/stepTotal);

    char perStep[24];
    if(ph == PHASE_INPUT) strcpy(perStep, "-");
    else snprintf(perStep, sizeof(perStep), "%.6f", totalMax/steps);

    CkPrintf("[TIMERS] %-14s %12.6f %12s %12.6f %8s\n",
             phaseName(ph), totalMax, perStep, totalMean, share);
  }
}

void Main::niceExit(){
  CkPrintf("[Main] graceful exit\n");
  CkExit();
}

void Main::quiescence(){
  CkPrintf("[Main] Quiescence detected!\n");
  treePieceProxy.quiescence();
  dataManagerProxy.quiescence();
}

void Main::quiescenceExit(){
  numQuiescenceRecvd++;
  if(numQuiescenceRecvd == 2){
    CkExit();
  }
}

string NodeTypeString[] = { 
  "Invalid",
  "Internal",
  "Bucket",
  "EmptyBucket",
  "Boundary",
  "Remote",
  "RemoteBucket",
  "RemoteEmptyBucket"
};

void Main::usage(){
  map<string,string> usage;
  usage["in"] = "input file";
  usage["format"] = "input format: 'csv' (id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z) "
                    "or 'binary' (./plummer, ./gen). Default: inferred from the extension";
  usage["ppc"] = "particles per chare";
  usage["b"] = "particles per bucket (leaf)";
  usage["theta"] = "opening angle";
  usage["G"] = "gravitational constant, in the units of the input dataset";
  usage["killat"] = "num single steps";
  usage["chunkDepth"] = "when fetching remote data, what depth of subtree to fetch";
  usage["yield"] = "how many buckets to process before yielding processor";
  usage["output"] = "write ParaView snapshots with this path prefix, giving "
                    "<prefix>.pvd to open; omit for no output";
  usage["outputfreq"] = "write a snapshot every Nth step (default 1)";
  usage["timers"] = "per-phase timing: 'all', 'none', or a comma-separated list of "
                    "input,decomposition,treebuild,traversal,integration,output,"
                    "loadbalancing,other,step";
  usage["ppc"] = "particleschare";


  map<string,string>::iterator it;
  CkPrintf("Usage: ./charmrun +p<numproc> ./barnes <options>\n");
  CkPrintf("Options take the form -<name>=<value>\n");
  CkPrintf("Options are described below:\n");
  for(it = usage.begin(); it != usage.end(); it++){
    CkPrintf("%s : %s\n", it->first.c_str(), it->second.c_str());
  }
}

#include "barnes.def.h"
