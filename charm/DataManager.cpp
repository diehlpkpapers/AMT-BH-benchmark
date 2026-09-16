#include "DataManager.h"
#include "Reduction.h"
#include "defines.h"
#include "Messages.h"
#include "Parameters.h"

#include "Worker.h"
#include "TreePiece.h"

#include "Request.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;
extern CProxy_TreePiece treePieceProxy;
extern CProxy_Main mainProxy;
extern Parameters globalParams;

void copyMomentsToNode(Node<ForceData> *node, const MomentsExchangeStruct &mes){
  CkAssert(node->getKey() == mes.key);

  node->data.moments = mes.moments;
  node->data.box = mes.box;
  NodeType type = mes.type;

  // A node built here over this PE's particles keeps a pointer into
  // myParticles even when none of them fall inside it -- an empty range, but a
  // non-NULL one. Once the owner's moments say the node is really a bucket
  // elsewhere, that pointer is stale, and Traversal::processLeaf reads a
  // non-NULL particle pointer as "the particles are already in hand" and skips
  // the fetch, silently dropping every one of them from the interaction.
  // Clear it, exactly as Node::deserialize does on the other path that turns a
  // node remote.
  node->setParticles(NULL,0);
  node->setType(Node<ForceData>::makeRemote(type));
}

DataManager::DataManager() : 
  activeBins(true),
  numTreePieces(1),
  tpMinParticles(0),
  tpMaxParticles(0),
  firstSplitterRound(false),
  decompIterations(0),
  iteration(0),
  haveRanges(false),
  keyRanges(NULL),
  rangeMsg(NULL),
  numLocalTreePieces(-1),
  doneTreeBuild(false),
  treeMomentsReady(false),
  numTreePiecesDoneTraversals(0),
  csvRangeCounts(NULL),
  warnedTreePieceBudget(false),
  prevIterationStart(0.0)
{
#ifdef STATISTICS
  numInteractions[0] = 0;
  numInteractions[1] = 0;
  numInteractions[2] = 0;
#endif
  avgIterationRuntime = 0.0;
  savedEnergy = 0.0;
  timers.init(globalParams.iterations, globalParams.timerMask);
}

void DataManager::loadParticles(const CkCallback &cb){
  numRankBits = LOG_BRANCH_FACTOR;
  timers.start(PHASE_INPUT);
  if(globalParams.inputFormat == INPUT_CSV) loadParticlesCsv(cb);
  else loadParticlesBinary(cb);
}

void DataManager::loadParticlesBinary(const CkCallback &cb){
  const char *fname = globalParams.filename.c_str();
  int npart = globalParams.numParticles;

  std::ifstream partFile;
  partFile.open(fname, ios::in | ios::binary);
  CkAssert(partFile.is_open());

  int offset = 0; 
  int myid = CkMyPe();
  int npes = CkNumPes();

  int avgParticlesPerPE = npart/npes;
  int rem = npart-npes*avgParticlesPerPE;
  if(myid < rem){
    avgParticlesPerPE++;
    offset = myid*avgParticlesPerPE;
  }
  else{
    offset = myid*avgParticlesPerPE+rem;
  }
  myNumParticles = avgParticlesPerPE;
  offset *= SIZE_PER_PARTICLE;
  offset += PREAMBLE_SIZE;

  myParticles.reserve(myNumParticles);
  myParticles.length() = myNumParticles;

  partFile.clear();
  partFile.seekg(offset,ios::beg);
  if(partFile.fail()){
    std::ostringstream oss;
    oss << "couldn't seek to position " << offset << " on PE " << CkMyPe() << " position " << partFile.tellg() << endl;
    CkAbort("%s", oss.str().c_str());
  }
  unsigned int numParticlesDone = 0;

  BoundingBox myBox;

  Real tmp[REALS_PER_PARTICLE];
  while(numParticlesDone < myNumParticles && !partFile.eof()){
    partFile.read((char *)tmp, SIZE_PER_PARTICLE);
    myParticles[numParticlesDone].position.x = tmp[0];
    myParticles[numParticlesDone].position.y = tmp[1];
    myParticles[numParticlesDone].position.z = tmp[2];
    myParticles[numParticlesDone].velocity.x = tmp[3];
    myParticles[numParticlesDone].velocity.y = tmp[4];
    myParticles[numParticlesDone].velocity.z = tmp[5];
    myParticles[numParticlesDone].mass = tmp[6];

    myParticles[numParticlesDone].acceleration.x = 0.0;
    myParticles[numParticlesDone].acceleration.y = 0.0;
    myParticles[numParticlesDone].acceleration.z = 0.0;
    myParticles[numParticlesDone].potential = 0.0;
    myBox.grow(myParticles[numParticlesDone].position);

    numParticlesDone++;
  }

  CkAssert(numParticlesDone == myNumParticles);
  myBox.numParticles = myNumParticles;

  partFile.close();

  timers.stop(PHASE_INPUT);
  contribute(sizeof(BoundingBox),&myBox,BoundingBoxGrowReductionType,cb);
}

// Positions `f` at the start of the first record at or after byte `start`.
// A record that straddles a byte-range boundary belongs to the range its
// first byte falls in, so a PE whose range begins mid-line discards that
// line -- its owner will read it. Returns the byte offset arrived at, or
// `fileSize` if the seek ran off the end.
static long seekToRecordStart(std::ifstream &f, long start, long dataOffset,
                              long fileSize){
  if(start <= dataOffset){
    f.clear();
    f.seekg(dataOffset, ios::beg);
    return dataOffset;
  }
  if(start >= fileSize) return fileSize;

  // If the byte before `start` ends a line then `start` is already a record
  // boundary; otherwise skip forward over the remainder of the straddling one.
  f.clear();
  f.seekg(start-1, ios::beg);
  char c = '\0';
  if(!f.get(c)) return fileSize;
  if(c == '\n') return start;

  std::string discard;
  if(!std::getline(f, discard)) return fileSize;
  long pos = (long)f.tellg();
  return pos < 0 ? fileSize : pos;
}

// Reads one record into `p`. Returns false for a blank line, which the caller
// skips without counting.
static bool parseCsvRecord(const std::string &line, Particle &p){
  if(isBlankCsvLine(line)) return false;

  Real f[CSV_NUM_FIELDS];
  if(!globalParams.csv.parseRow(line, f)){
    std::ostringstream oss;
    oss << "PE " << CkMyPe() << ": malformed CSV record: " << line << endl;
    CkAbort("%s", oss.str().c_str());
  }

  p.mass = f[CSV_MASS];
  p.position = Vector3D<Real>(f[CSV_POS_X], f[CSV_POS_Y], f[CSV_POS_Z]);
  p.velocity = Vector3D<Real>(f[CSV_VEL_X], f[CSV_VEL_Y], f[CSV_VEL_Z]);
  p.acceleration = Vector3D<Real>(0.0);
  p.potential = 0.0;
  return true;
}

// CSV load, phase 1 of 2.
//
// Unlike the binary format, a text file gives no way to compute the byte
// offset of record i, so a PE cannot seek straight to the slice it should own.
// Each PE instead takes an equal *byte* range and counts the records that
// begin in it. Rows vary in width -- if nothing else, the id column grows a
// digit every power of ten -- so those counts are not equal, and phase 2 uses
// them to recover an exactly even split.
void DataManager::loadParticlesCsv(const CkCallback &cb){
  loadParticlesCb = cb;

  const long dataOffset = globalParams.csv.dataOffset;
  const long fileSize = globalParams.inputFileSize;
  const long long dataBytes = (long long)(fileSize - dataOffset);
  const int npes = CkNumPes();
  const int myid = CkMyPe();

  const long myStart = dataOffset + (long)(dataBytes*myid/npes);
  const long myEnd = dataOffset + (long)(dataBytes*(myid+1)/npes);

  std::ifstream partFile(globalParams.filename.c_str(), ios::in | ios::binary);
  CkAssert(partFile.is_open());

  long records = 0;
  long pos = seekToRecordStart(partFile, myStart, dataOffset, fileSize);
  std::string line;
  while(pos < myEnd && std::getline(partFile, line)){
    if(!isBlankCsvLine(line)) records++;
    long next = (long)partFile.tellg();
    pos = (next < 0) ? fileSize : next;   // tellg() is -1 once eof is set
  }
  partFile.close();

  // Sum a one-hot vector so that every PE ends up with every PE's count, and
  // therefore with the same prefix sums. npes ints; this is not the hot path.
  delete[] csvRangeCounts;
  csvRangeCounts = new int[npes];
  for(int i = 0; i < npes; i++) csvRangeCounts[i] = 0;
  csvRangeCounts[myid] = (int)records;

  contribute(npes*sizeof(int), csvRangeCounts, CkReduction::sum_int,
             CkCallback(CkIndex_DataManager::receiveCsvCounts(NULL), thisProxy));
}

// CSV load, phase 2 of 2.
//
// `msg` holds record counts per byte range, so prefix[j] is the global index
// of the first record in PE j's byte range. This PE reads the global range it
// is owed -- an exactly even slice -- by seeking into whichever byte range
// contains its first record and reading straight through, past range
// boundaries as needed. Both ranges are about N/P long, so this touches at
// most a couple of them: one extra scan, not a redistribution.
void DataManager::receiveCsvCounts(CkReductionMsg *msg){
  const int npes = CkNumPes();
  const int myid = CkMyPe();
  const int *counts = (const int *)msg->getData();

  CkVec<long> prefix;
  prefix.resize(npes+1);
  prefix[0] = 0;
  for(int i = 0; i < npes; i++) prefix[i+1] = prefix[i] + counts[i];
  const long total = prefix[npes];
  delete msg;

  if(total != (long)globalParams.numParticles){
    std::ostringstream oss;
    oss << "PE " << CkMyPe() << ": counted " << total
        << " CSV records but the scan on PE 0 found "
        << globalParams.numParticles << endl;
    CkAbort("%s", oss.str().c_str());
  }

  const long myFirst = (long)(total*(long long)myid/npes);
  const long myLast = (long)(total*(long long)(myid+1)/npes);
  myNumParticles = (int)(myLast - myFirst);

  myParticles.reserve(myNumParticles);
  myParticles.length() = myNumParticles;

  BoundingBox myBox;

  if(myNumParticles > 0){
    // The byte range holding record myFirst.
    int range = 0;
    while(range+1 < npes && prefix[range+1] <= myFirst) range++;

    const long dataOffset = globalParams.csv.dataOffset;
    const long fileSize = globalParams.inputFileSize;
    const long long dataBytes = (long long)(fileSize - dataOffset);
    const long rangeStart = dataOffset + (long)(dataBytes*range/npes);

    std::ifstream partFile(globalParams.filename.c_str(), ios::in | ios::binary);
    CkAssert(partFile.is_open());
    seekToRecordStart(partFile, rangeStart, dataOffset, fileSize);

    long toSkip = myFirst - prefix[range];
    std::string line;
    while(toSkip > 0 && std::getline(partFile, line)){
      if(!isBlankCsvLine(line)) toSkip--;
    }
    if(toSkip > 0){
      CkAbort("ran out of CSV records while seeking to this PE's slice\n");
    }

    int done = 0;
    while(done < myNumParticles && std::getline(partFile, line)){
      if(!parseCsvRecord(line, myParticles[done])) continue;
      myBox.grow(myParticles[done].position);
      done++;
    }
    if(done != myNumParticles){
      CkAbort("ran out of CSV records while reading this PE's slice\n");
    }
    partFile.close();
  }

  myBox.numParticles = myNumParticles;

  delete[] csvRangeCounts;
  csvRangeCounts = NULL;

  timers.stop(PHASE_INPUT);
  contribute(sizeof(BoundingBox), &myBox, BoundingBoxGrowReductionType,
             loadParticlesCb);
}

void DataManager::hashParticleCoordinates(const OrientedBox<Real> &universe){
  Key prepend;
  prepend = 1L;
  prepend <<= (TREE_KEY_BITS-1);

  Real xsz = universe.greater_corner.x-universe.lesser_corner.x;
  Real ysz = universe.greater_corner.y-universe.lesser_corner.y;
  Real zsz = universe.greater_corner.z-universe.lesser_corner.z;

  for(unsigned int i = 0; i < myNumParticles; i++){
    Particle *p = &(myParticles[i]);
    Key xint = ((Key) (((p->position.x-universe.lesser_corner.x)*(BOXES_PER_DIM*1.0))/xsz)); 
    Key yint = ((Key) (((p->position.y-universe.lesser_corner.y)*(BOXES_PER_DIM*1.0))/ysz)); 
    Key zint = ((Key) (((p->position.z-universe.lesser_corner.z)*(BOXES_PER_DIM*1.0))/zsz)); 

    Key mask = Key(0x1);
    Key k = Key(0x0);
    int shiftBy = 0;
    for(int j = 0; j < BITS_PER_DIM; j++){
      k |= ((zint & mask) <<  shiftBy);
      k |= ((yint & mask) << (shiftBy+1));
      k |= ((xint & mask) << (shiftBy+2));
      mask <<= 1;
      // minus 1 because mask itself has shifted
      // left by one position
      shiftBy += (NDIMS-1);
    }
    k |= prepend; 
    myParticles[i].key = k;
  }
}

void DataManager::decompose(const BoundingBox &universe){
  timers.setStep(iteration);
  timers.start(PHASE_STEP);
  timers.start(PHASE_DECOMPOSITION);

  hashParticleCoordinates(universe.box);
  myParticles.quickSort();

  if(CkMyPe()==0){
    float memMB = (1.0*CmiMemoryUsage())/(1<<20);
    ostringstream oss; 
#ifdef STATISTICS
    CkPrintf("(%d) prev time %g s\n", CkMyPe(), CmiWallTimer()-prevIterationStart);
    CkPrintf("(%d) start iteration %d\n", CkMyPe(), iteration);
    CkPrintf("(%d) mem %.2f MB\n", CkMyPe(), memMB);
    CkPrintf("(%d) univ %g %g %g %g %g %g\n",
              CkMyPe(),
              universe.box.lesser_corner.x,
              universe.box.lesser_corner.y,
              universe.box.lesser_corner.z,
              universe.box.greater_corner.x,
              universe.box.greater_corner.y,
              universe.box.greater_corner.z);
    // E_T = E_K + E_P should hold steady across the run; that is the
    // whitepaper's correctness check (Section 6).
    //
    // These energies were computed by the previous step's kickDriftKick, from
    // the state at the *start* of that step, so they are labelled with that
    // step's index. Before the first advance there is nothing to report yet.
    if(iteration > 0){
      CkPrintf("[ENERGY] step %d E_K %.10g E_P %.10g E_T %.10g\n",
                iteration-1,
                universe.kineticEnergy,
                universe.potentialEnergy,
                universe.totalEnergy());
    }
#endif

    avgIterationRuntime += (CmiWallTimer()-prevIterationStart);
    prevIterationStart = CkWallTimer();
  }

  numTreePieces = 1;
  initHistogramParticles();
  sendHistogram();
}

void DataManager::initHistogramParticles(){
  int rootDepth = 0;
  
  sortingRoot = new Node<NodeDescriptor>(Key(1),
                         rootDepth,
                         myParticles.getVec(),
                         myNumParticles);
  activeBins.addNewNode(sortingRoot);

  // don't access myParticles through ckvec after this
  // anyway. these must be reset before this DM starts
  // to receive submitted particles from TPs placed on it
  myNumParticles = 0;
  myParticles.length() = 0;
}

void DataManager::sendHistogram(){

  CkCallback cb(CkIndex_DataManager::receiveHistogram(NULL),0,this->thisgroup);
  contribute(sizeof(NodeDescriptor)*activeBins.getNumCounts(),activeBins.getCounts(),NodeDescriptorReductionType,cb);
  activeBins.reset();
}

// executed on PE 0
void DataManager::receiveHistogram(CkReductionMsg *msg){

  int numRecvdBins = msg->getSize()/sizeof(NodeDescriptor);
  NodeDescriptor *descriptors = (NodeDescriptor *)msg->getData();

  // XXX remove this and make a refine function for ActiveBinInfo
  CkVec<int> binsToRefine;

  binsToRefine.reserve(numRecvdBins);
  binsToRefine.length() = 0;

  int particlesHistogrammed = 0;

  CkVec<pair<Node<NodeDescriptor>*,bool> > *active = activeBins.getActive();
  CkAssert(numRecvdBins == active->length());

  // How many leaves the refinement needs is a property of the input: the
  // splitters cut the Morton key space at midpoints rather than at medians, so
  // a clustered region keeps splitting long after an even one has stopped.
  // -p is therefore a budget, not a prediction. Running out of it used to
  // abort the run outright; instead, stop subdividing and let the remaining
  // oversized bins through. The result is a coarser, less even decomposition
  // -- which is worth a warning -- but it is a correct one, and it costs no
  // accuracy: bucket refinement in buildTree() is driven by particle count,
  // not by TreePiece boundaries.
  int overBudget = 0;

  // A bin whose particles all share one Morton key cannot be split -- every
  // particle lands in the same child at every depth, forever. Splitting it
  // anyway burns one leaf of the -p budget per level and walks the refinement
  // off the bottom of the 63-bit key onto an assertion, which is what any
  // input with a clump of coincident bodies used to do. Such a bin is settled
  // whatever its count.
  for(int i = 0; i < numRecvdBins; i++){
    bool wantsRefine =
        descriptors[i].numParticles > (Real)(DECOMP_TOLERANCE*globalParams.ppc) &&
        descriptors[i].smallestKey != descriptors[i].largestKey;
    bool canRefine =
        numTreePieces + (BRANCH_FACTOR-1) <= globalParams.numTreePieces;

    if(wantsRefine && canRefine){
      // need to refine this bin (partition)
      binsToRefine.push_back(i);
      numTreePieces += (BRANCH_FACTOR-1);
    }
    else{
      if(wantsRefine) overBudget++;
      Node<NodeDescriptor> *nd = (*active)[i].first;
      nd->data = descriptors[i];
    }

    particlesHistogrammed += descriptors[i].numParticles;
  }

  if(overBudget > 0 && !warnedTreePieceBudget){
    CkPrintf("[0] warning: ran out of TreePieces at %d (-p); %d bin%s left above "
             "the %g-particle target, so the decomposition is coarser than asked "
             "for. Raise -p to remove this.\n",
             globalParams.numTreePieces, overBudget, overBudget == 1 ? "" : "s",
             (double)(DECOMP_TOLERANCE*globalParams.ppc));
    warnedTreePieceBudget = true;
  }

  // The loop above stops as soon as every bin is under the target, which
  // normally leaves a good part of the -p budget unspent: 1277 TreePieces of
  // 2016 on 100k Plummer bodies. That is not just waste. The leaves are handed
  // to TreePieces 0..numTreePieces-1, and Charm++'s default map gives each PE
  // a contiguous *block* of array indices, so an unspent top third of the
  // array is an idle top third of the machine: at 4 PEs one of them got no
  // particles at all, at 8 PEs two did.
  //
  // So spend the rest of the budget. Once no bin wants refining on its own
  // account, split the heaviest bins that can still be split until the leaf
  // count reaches -p exactly. The decomposition comes out finer and more even
  // rather than coarser, and there are no leftover TreePieces for the map to
  // fill a PE with.
  //
  // Cost is one more histogram round in the usual case: the leaf count at most
  // doubles per round, and the budget is under 2x what the target refinement
  // already produced.
  //
  // A bin whose particles all share one key cannot be split -- every particle
  // lands in the same child, at every depth -- so those are not candidates. If
  // too few candidates remain, the budget stays partly unspent; the balance
  // report prints how many TreePieces were used of how many.
  if(binsToRefine.length() == 0 && numTreePieces < globalParams.numTreePieces){
    vector<pair<int,int> > candidates;   // (-numParticles, bin), heaviest first
    for(int i = 0; i < numRecvdBins; i++){
      if(descriptors[i].smallestKey != descriptors[i].largestKey){
        candidates.push_back(make_pair(-descriptors[i].numParticles,i));
      }
    }

    int room = globalParams.numTreePieces - numTreePieces;
    int take = (int)candidates.size() < room ? (int)candidates.size() : room;

    partial_sort(candidates.begin(),candidates.begin()+take,candidates.end());

    // Back into bin order, so the active bins stay in tree order as they do on
    // every other round.
    vector<int> extra;
    for(int i = 0; i < take; i++) extra.push_back(candidates[i].second);
    sort(extra.begin(),extra.end());

    for(int i = 0; i < take; i++) binsToRefine.push_back(extra[i]);
    numTreePieces += take*(BRANCH_FACTOR-1);
  }

  int numBinsToRefine = binsToRefine.length();

  if(numBinsToRefine > 0){
    SplitterMsg *m = new (numBinsToRefine,0) SplitterMsg;
    memcpy(m->splitBins,binsToRefine.getVec(),sizeof(int)*numBinsToRefine);
    m->nSplitBins = numBinsToRefine; 
    thisProxy.receiveSplitters(m);
    decompIterations++;
  }
  else{
    // One bin for the whole universe means every body hashed to the same Morton
    // key: they all sit inside a single cell of the 2^BITS_PER_DIM grid, which
    // is to say at the same position to the only resolution the keys have.
    // There is no decomposition to make, nothing the tree can separate, and
    // every separation in the force kernel is zero. Left alone the run does
    // not fail, it stalls -- the one TreePiece that has particles never gets a
    // bucket, and quiescence is detected with no forces computed.
    if(numRecvdBins == 1 && descriptors[0].numParticles > 1 &&
       descriptors[0].smallestKey == descriptors[0].largestKey){
      ostringstream oss;
      oss << "all " << descriptors[0].numParticles << " bodies share one "
          << "position to the resolution of the " << BITS_PER_DIM
          << "-bit-per-dimension Morton key; there is nothing to decompose"
          << endl;
      CkAbort("%s", oss.str().c_str());
    }

    // Every bin is a leaf and every leaf becomes one TreePiece, so the bin
    // counts are the TreePiece counts. Recorded here, while the global
    // histogram is still in hand, for the balance line to report later.
    tpMinParticles = tpMaxParticles = descriptors[0].numParticles;
    for(int i = 1; i < numRecvdBins; i++){
      if(descriptors[i].numParticles < tpMinParticles)
        tpMinParticles = descriptors[i].numParticles;
      if(descriptors[i].numParticles > tpMaxParticles)
        tpMaxParticles = descriptors[i].numParticles;
    }

    // create tree pieces and send proxy
    #ifdef STATISTICS
    CkPrintf("[0] decomp done after %d iterations used treepieces %d\n", decompIterations, numTreePieces);
    #endif
    decompIterations = 0;
    
    keyRanges = new Key[numTreePieces*2];
    // so that by the time tree pieces start submitting
    // particles (which can only happen after the flushParticles()
    // below, we have the right count of local tree pieces
    senseTreePieces();
    flushParticles();
    // PE 0 sets ranges in sendParticlesToTreePiece
    haveRanges = true;

    int numKeys = numTreePieces*2;

    RangeMsg *rmsg = new (numKeys) RangeMsg;
    rmsg->numTreePieces = numTreePieces;
    memcpy(rmsg->keys,keyRanges,sizeof(Key)*numKeys);
    thisProxy.sendParticles(rmsg);
  }

  delete msg;
}

void DataManager::flushParticles(){
  ParticleFlushWorker pfw(this);
  scaffoldTrav.preorderTraversal(sortingRoot,&pfw);

  int numUsefulTreePieces = pfw.getNumLeaves(); 

  for(int i = numUsefulTreePieces; i < globalParams.numTreePieces; i++){
    treePieceProxy[i].receiveParticles();
  }

  // done with sorting tree; delete
  FreeTreeWorker<NodeDescriptor> freeWorker;
  scaffoldTrav.postorderTraversal(sortingRoot,&freeWorker);

  delete sortingRoot;
  sortingRoot = NULL;
}

void DataManager::receiveSplitters(SplitterMsg *msg){

  int numRefineBins = msg->nSplitBins;

  // process bins to refine
  activeBins.processRefine(msg->splitBins,msg->nSplitBins);

  // We traverse the final tree to flush particles to 
  // appropriate tree pieces

  // here, we will know of bins that have not
  // been refined or deleted in the present 
  // iteration; we can send particles to these
  // CkVec<Node<NodeDescriptor>*> &unrefined = activeBins.getUnrefined();

  sendHistogram();
  delete msg;
}

void DataManager::sendParticlesToTreePiece(Node<NodeDescriptor> *nd, int tp) {
  CkAssert(nd->getNumChildren() == 0);
  int np = nd->getNumParticles();

  if(np > 0){
    ParticleMsg *msg = new (np,0) ParticleMsg;
    memcpy(msg->part, nd->getParticles(), sizeof(Particle)*np);
    msg->numParticles = np;
    treePieceProxy[tp].receiveParticles(msg);
  }
  else{
    treePieceProxy[tp].receiveParticles();
  }

  // only PE 0 has the correct ranges
  if(CkMyPe() == 0){
    if(nd->data.numParticles > 0){
      CkAssert(nd->data.smallestKey <= nd->data.largestKey);
    } else {
      CkAssert(nd->data.smallestKey == nd->data.largestKey);
    }

    keyRanges[(tp<<1)] = nd->data.smallestKey;
    keyRanges[(tp<<1)+1] = nd->data.largestKey;

  }
}

void DataManager::sendParticles(RangeMsg *msg){

  if(CkMyPe() != 0){
    numTreePieces = msg->numTreePieces;
    keyRanges = msg->keys;
    haveRanges = true;
    // delete this later
    rangeMsg = msg;
    flushParticles();

    senseTreePieces();

    if(submittedParticles.length() == numLocalTreePieces){
      processSubmittedParticles();
    }
  }
  else{
    CkAssert(numTreePieces == msg->numTreePieces);
    CkAssert(haveRanges);
    delete msg;
  }

  // there are tree piece on this PE
  // these will eventually receive their respective
  // particles and submit them to the DM. Also, the
  // DM will receive the rangeKeys from the decomposition
  // leader (i.e. DM on PE 0) When all particles and 
  // rangeKeys have been received, the DM begins tree
  // construction
}

void DataManager::senseTreePieces(){
  localTreePieces.reset();
  CkLocMgr *mgr = treePieceProxy.ckLocMgr();
  mgr->iterate(localTreePieces);
  numLocalTreePieces = localTreePieces.count;
}

void DataManager::submitParticles(CkVec<ParticleMsg*> *vec, int numParticles, TreePiece * tp, Key smallestKey, Key largestKey){ 
  submittedParticles.push_back(TreePieceDescriptor(vec,numParticles,tp,tp->getIndex(),smallestKey,largestKey));
  myNumParticles += numParticles;
  if(submittedParticles.length() == numLocalTreePieces &&
     haveRanges){
    processSubmittedParticles();
  }
}

void DataManager::processSubmittedParticles(){
  int offset = 0;

  submittedParticles.quickSort();
  
  myParticles.resize(myNumParticles);

  for(int i = 0; i < submittedParticles.length(); i++){
    TreePieceDescriptor &descr = submittedParticles[i];
    CkVec<ParticleMsg*> *vec = descr.vec;
    for(int j = 0; j < vec->length(); j++){
      ParticleMsg *msg = (*vec)[j];
      memcpy(myParticles.getVec()+offset,msg->part,sizeof(Particle)*msg->numParticles);
      offset += msg->numParticles;
      delete msg;
    }
  }

  myParticles.quickSort();

  if(iteration == 0) reportBalance();

  // The particles this PE will own for the rest of the step are now in hand
  // and in key order; everything from here to treeReady() is tree construction.
  timers.stop(PHASE_DECOMPOSITION);
  timers.start(PHASE_TREEBUILD);

  buildTree();
  // add dummy tree piece whose index is larger than
  // that of all others. this is required to mark the
  // boundary of nodes/particles owned by this PE.
  submittedParticles.push_back(TreePieceDescriptor(globalParams.numTreePieces));

  // makeMoments also sends out requests for moments 
  // of remote nodes
  makeMoments();

  doneTreeBuild = true;

  // are all particles local to this PE? 
  if(root != NULL && root->getType() == Internal){
    passMomentsUpward(root);
  }

  flushMomentRequests();
}

// How the decomposition actually came out, reduced once after the first
// step's exchange. Cheap, and the only way to see the difference between a
// decomposition that is merely correct and one that is worth putting on more
// than one PE.
//
// Slot p carries PE p's particle count; the slot past the end carries the
// number of TreePieces the refinement used, which only PE 0 knows -- every
// other PE contributes 0 there, so the sum leaves PE 0's value standing.
void DataManager::reportBalance(){
  int n = CkNumPes()+3;
  int *counts = new int[n];
  memset(counts,0,sizeof(int)*n);
  counts[CkMyPe()] = myNumParticles;
  // The last three slots describe the TreePieces rather than the PEs, and only
  // PE 0 knows them -- it is the one that ran the histogram. Everyone else
  // leaves zeros there, so the sum passes PE 0's values through untouched.
  if(CkMyPe() == 0){
    counts[CkNumPes()] = numTreePieces;
    counts[CkNumPes()+1] = tpMinParticles;
    counts[CkNumPes()+2] = tpMaxParticles;
  }

  CkCallback cb(CkIndex_Main::reportBalance(NULL),mainProxy);
  contribute(sizeof(int)*n,counts,CkReduction::sum_int,cb);
  delete[] counts;
}

void DataManager::buildTree(){

  int rootDepth = 0;
  root = new Node<ForceData>(Key(1),rootDepth,myParticles.getVec(),myNumParticles);
  root->setOwners(0,numTreePieces-1);
  nodeTable[Key(1)] = root;
  if(myNumParticles == 0){
    return;
  }

  OwnershipActiveBinInfo<ForceData> abi(keyRanges);
  abi.addNewNode(root);
  int numFatNodes = 1;

  int limit = ((Real)globalParams.ppb*BUCKET_TOLERANCE);

  CkVec<int> refines;

  while(numFatNodes > 0){
    abi.reset();
    refines.length() = 0;

    CkVec<std::pair<Node<ForceData>*,bool> > *active = abi.getActive();
    // discard node when:
    // 1. ownerEnd of node is < curTP
    // 2. if not 1, check for numparticles in node 
    // if ownerStart of node is > curTP, curTP = curTP->next
    // when a node its split, 
    for(int i = 0; i < active->length(); i++){
      Node<ForceData> *node = (*active)[i].first;

      // Particles sharing a key cannot be told apart by refining -- they land
      // in the same child at every depth -- so an oversized node made of them
      // is as small as it will ever get. Without this the bucket criterion
      // never lets go and the refinement runs off the bottom of the key: any
      // input with more than ppb bodies at one position (a quantised dataset,
      // a generator with a singular core) aborted here on the depth assert.
      // The node stays a bucket above its target size, which costs
      // particle-particle work but is correct.
      //
      // The ownership criterion is safe as it stands: refining narrows the
      // node's key range whatever the particles do, and no two TreePieces
      // share a key, so a node's owners always resolve to one.
      Particle *particles = node->getParticles();
      int numParticles = node->getNumParticles();
      bool separable =
          numParticles > 0 && particles[0].key != particles[numParticles-1].key;

      if((node->getOwnerEnd() > node->getOwnerStart()) ||
         (numParticles > limit && separable)){
        refines.push_back(i);
      }
    }

    abi.processRefine(refines.getVec(), refines.length());
    numFatNodes = abi.getNumCounts();
  }

}

void DataManager::makeMoments(){
  if(root == NULL) return;

  MomentsWorker mw(submittedParticles,
                   nodeTable,
                   myBuckets
                   );
  fillTrav.postorderTraversal(root,&mw);
}

Node<ForceData> *DataManager::lookupNode(Key k){
  map<Key,Node<ForceData>*>::iterator it;
  it = nodeTable.find(k);
  if(it == nodeTable.end()) return NULL;
  else return it->second;
}

void DataManager::requestMoments(Key k, int replyTo){
  pendingMoments[k].push_back(replyTo);
  TB_DEBUG("(%d) received requestMoments %lu from pe %d doneTreeBuild %d\n", 
          CkMyPe(), k, replyTo, doneTreeBuild);

  if(doneTreeBuild){    
    Node<ForceData> *node = lookupNode(k);
    if(node == NULL){
      CkPrintf("(%d) recvd request from %d for moments %lu\n", CkMyPe(), replyTo, k);
      CkAbort("bad request\n");
    }
    bool ready = node->allChildrenMomentsReady();
    TB_DEBUG("(%d) node %lu ready %d\n", CkMyPe(), k, ready);

    if(ready){
      map<Key,CkVec<int> >::iterator it = pendingMoments.find(k);
      CkVec<int> &requestors = it->second;
      respondToMomentsRequest(node,requestors);
      pendingMoments.erase(it);
    }
  }
}

void DataManager::flushMomentRequests(){
  CkAssert(doneTreeBuild);
  map<Key,CkVec<int> >::iterator it;
  for(it = pendingMoments.begin(); it != pendingMoments.end();){
    Key k = it->first;
    Node<ForceData> *node = lookupNode(k);
    CkAssert(node != NULL);
    if(node->allChildrenMomentsReady()){
      CkVec<int> &requestors = it->second;
      respondToMomentsRequest(node,requestors);
      map<Key,CkVec<int> >::iterator kill = it;
      ++it;
      pendingMoments.erase(kill);
    }
    else{
      ++it;
    }
  }
}

void DataManager::respondToMomentsRequest(Node<ForceData> *node, CkVec<int> &replyTo){
  for(int i = 0; i < replyTo.length(); i++){
    MomentsMsg *m = new (NUM_PRIORITY_BITS) MomentsMsg(node);
    *(int *)CkPriorityPtr(m) = RECV_MOMENTS_PRIORITY;
    CkSetQueueing(m,CK_QUEUEING_IFIFO);
    TB_DEBUG("(%d) responding to %d with node %lu\n", CkMyPe(), replyTo[i], node->getKey());
    thisProxy[replyTo[i]].receiveMoments(m);
  }
  replyTo.length() = 0;
}

void DataManager::receiveMoments(MomentsMsg *msg){
  Node<ForceData> *node = lookupNode(msg->data.key);
  CkAssert(node != NULL);

  // update moments of leaf and pass these on 
  // to parent recursively; if there are requests
  // for these nodes, respond to them
  updateLeafMoments(node,msg->data);
   
  delete msg;
}

void DataManager::updateLeafMoments(Node<ForceData> *node, MomentsExchangeStruct &data){
  copyMomentsToNode(node,data);
  TB_DEBUG("(%d) updateLeafMoments %lu\n", CkMyPe(), node->getKey());
  passMomentsUpward(node);
}

void DataManager::passMomentsUpward(Node<ForceData> *node){
  TB_DEBUG("(%d) passUp %lu\n", CkMyPe(), node->getKey());
  map<Key,CkVec<int> >::iterator it = pendingMoments.find(node->getKey());
  if(it != pendingMoments.end()){
    CkVec<int> &requestors = it->second;
    respondToMomentsRequest(node,requestors);
    pendingMoments.erase(it);
  }

  Node<ForceData> *parent = node->getParent();
  if(parent == NULL){
    CkAssert(node->getKey() == Key(1));
    treeReady();
  }else{
    parent->childMomentsReady();
    TB_DEBUG("[%d] parent %lu children ready %d\n", CkMyPe(), parent->getKey(), parent->getNumChildrenMomentsReady());
    if(parent->allChildrenMomentsReady()){
      parent->getMomentsFromChildren();
      parent->getOwnershipFromChildren();
      passMomentsUpward(parent);
    }
  }
}

// doneTreeBuild: built local tree and sent out requests for remote 

void DataManager::treeReady(){
  timers.stop(PHASE_TREEBUILD);
  timers.start(PHASE_TRAVERSAL);

  treeMomentsReady = true;
  flushBufferedRemoteDataRequests();
  startTraversal();
}

void DataManager::flushBufferedRemoteDataRequests(){
  CkAssert(treeMomentsReady);
  for(int i = 0; i < bufferedNodeRequests.length(); i++){
    RequestMsg *msg = bufferedNodeRequests[i];
    requestNode(msg);
  }
  for(int i = 0; i < bufferedParticleRequests.length(); i++){
    RequestMsg *msg = bufferedParticleRequests[i];
    requestParticles(msg);
  }
  bufferedNodeRequests.length() = 0;
  bufferedParticleRequests.length() = 0;
}

bool CompareNodePtrToKey(void *a, Key k){
  Node<ForceData> *node = *((Node<ForceData>**)a);
  return (Node<ForceData>::getParticleLevelKey(node) >= k);
}

void DataManager::startTraversal(){
  Node<ForceData> **bucketPtrs = myBuckets.getVec();
  submittedParticles[0].bucketStartIdx = 0;
  int start = 0;
  int end = myBuckets.length();

  if(end > 0){
    for(int i = 0; i < numLocalTreePieces-1; i++){
      TreePieceDescriptor &descr = submittedParticles[i];
      int bucketIdx = binary_search_ge<Node<ForceData>*>(descr.largestKey,bucketPtrs,start,end,CompareNodePtrToKey);
      descr.bucketEndIdx = bucketIdx;
      descr.owner->prepare(root,myBuckets.getVec(),descr.bucketStartIdx,descr.bucketEndIdx);
      int tpIndex = descr.owner->getIndex();
      treePieceProxy[tpIndex].startTraversal();
      submittedParticles[i+1].bucketStartIdx = bucketIdx;
      start = bucketIdx;
    }
    TreePieceDescriptor &descr = submittedParticles[numLocalTreePieces-1];
    descr.bucketEndIdx = myBuckets.length();
    descr.owner->prepare(root,myBuckets.getVec(),descr.bucketStartIdx,descr.bucketEndIdx);
    int tpIndex = descr.owner->getIndex();
    treePieceProxy[tpIndex].startTraversal();
  }
  else if(numLocalTreePieces > 0){
    for(int i = 0; i < numLocalTreePieces; i++){
      TreePieceDescriptor &descr = submittedParticles[i];
      descr.owner->prepare(root,myBuckets.getVec(),0,0);
      int tpIndex = descr.owner->getIndex();
      treePieceProxy[tpIndex].startTraversal();
    }
  }
  else{
    finishIteration();
  }
}

void DataManager::requestParticles(Node<ForceData> *leaf, CutoffWorker<ForceData> *worker, State *state, Traversal<ForceData> *traversal){
  Key key = leaf->getKey();
  Request &request = particleRequestTable[key];
  if(!request.sent){
    partReqs.incrRequests();

    if(leaf->isCached()) request.parentCached = true;
    else request.parentCached = false;

    RequestMsg *reqMsg = new (NUM_PRIORITY_BITS) RequestMsg(key,CkMyPe());
    *(int *)CkPriorityPtr(reqMsg) = REQUEST_PARTICLES_PRIORITY;
    CkSetQueueing(reqMsg,CK_QUEUEING_IFIFO);

    CkAssert(leaf->getOwnerStart() == leaf->getOwnerEnd());
    int owner = leaf->getOwnerStart();
    treePieceProxy[owner].requestParticles(reqMsg);
    request.sent = true;
    
    request.parent = leaf;
    RRDEBUG("(%d) REQUEST particles %lu from tp %d\n", 
            CkMyPe(), key, owner);
  }
  request.requestors.push_back(Requestor(worker,state,traversal,worker->getContext()));
  partReqs.incrDeliveries();
}

void DataManager::requestParticles(RequestMsg *msg){
  if(!treeMomentsReady){
    bufferedParticleRequests.push_back(msg);
    return;
  }

  RRDEBUG("(%d) REPLY particles key %lu to %d\n", 
          CkMyPe(), msg->key, msg->replyTo);

  map<Key,Node<ForceData>*>::iterator it = nodeTable.find(msg->key);
  CkAssert(it != nodeTable.end());
  Node<ForceData> *bucket = it->second;
  CkAssert(bucket->getType() == Bucket);

  Particle *data = bucket->getParticles();
  int np = bucket->getNumParticles();

  ParticleReplyMsg *pmsg = new (np,NUM_PRIORITY_BITS) ParticleReplyMsg;
  *(int *)CkPriorityPtr(pmsg) = RECV_PARTICLES_PRIORITY;
  CkSetQueueing(pmsg,CK_QUEUEING_IFIFO);

  pmsg->key = msg->key;
  pmsg->np = np;
  for(int i = 0; i < np; i++){
    pmsg->data[i] = data[i];
  }

  thisProxy[msg->replyTo].recvParticles(pmsg);
  delete msg;
}

void DataManager::requestNode(Node<ForceData> *leaf, CutoffWorker<ForceData> *worker, State *state, Traversal<ForceData> *traversal){
  Key key = leaf->getKey();
  Request &request = nodeRequestTable[key];
  if(!request.sent){
    nodeReqs.incrRequests();

    if(leaf->isCached()) request.parentCached = true;
    else request.parentCached = false;

    RequestMsg *reqMsg = new (NUM_PRIORITY_BITS) RequestMsg(key,CkMyPe());
    *(int *)CkPriorityPtr(reqMsg) = REQUEST_NODE_PRIORITY;
    CkSetQueueing(reqMsg,CK_QUEUEING_IFIFO);

    int numOwners = leaf->getOwnerEnd()-leaf->getOwnerStart()+1;
    int requestOwner = leaf->getOwnerStart()+(rand()%numOwners);
    RRDEBUG("(%d) REQUEST node %lu from tp %d\n", 
            CkMyPe(), key, requestOwner);
    treePieceProxy[requestOwner].requestNode(reqMsg);
    request.sent = true;
    request.parent = leaf;
  }
  request.requestors.push_back(Requestor(worker,state,traversal,worker->getContext()));
  nodeReqs.incrDeliveries();
}

void DataManager::requestNode(RequestMsg *msg){
  if(!treeMomentsReady){
    bufferedNodeRequests.push_back(msg);
    return;
  }

  RRDEBUG("(%d) REPLY node %lu to %d\n", 
          CkMyPe(), msg->key, msg->replyTo);


  map<Key,Node<ForceData>*>::iterator it = nodeTable.find(msg->key);
  CkAssert(it != nodeTable.end());
  Node<ForceData> *node = it->second;

  if(node->getNumChildren() == 0){
    CkPrintf("[%d] children of leaf node %lu type %s requested!\n", CkMyPe(), node->getKey(), NodeTypeString[node->getType()].c_str());
    CkAbort("Leaf children request\n");
  }

  TreeSizeWorker tsz(node->getDepth()+globalParams.cacheLineSize);
  fillTrav.topDownTraversal_local(node,&tsz);

  int nn = tsz.getNumNodes();

  NodeReplyMsg *nmsg = new (nn,NUM_PRIORITY_BITS) NodeReplyMsg;
  *(int *)CkPriorityPtr(nmsg) = RECV_NODE_PRIORITY;
  CkSetQueueing(nmsg,CK_QUEUEING_IFIFO);

  nmsg->key = msg->key;
  nmsg->nn = nn;

  Node<ForceData> *emptyBuf = nmsg->data;
  node->serialize(NULL,emptyBuf,globalParams.cacheLineSize);
  CkAssert(emptyBuf == nmsg->data+nn);

  thisProxy[msg->replyTo].recvNode(nmsg);

  delete msg;
}

void DataManager::recvParticles(ParticleReplyMsg *msg){
  map<Key,Request>::iterator it = particleRequestTable.find(msg->key);
  CkAssert(it != particleRequestTable.end());

  RRDEBUG("(%d) RECVD particle REPLY for key %lu\n", 
          CkMyPe(), msg->key);

  Request &req = it->second;
  CkAssert(req.requestors.length() > 0);
  CkAssert(req.sent);
  CkAssert(req.msg == NULL);

  req.msg = msg;
  req.data = msg->data;
  
  // attach particles to bucket in tree 
  Node<ForceData> *leaf = req.parent;
  CkAssert(leaf != NULL);
  CkAssert(leaf->getType() == RemoteBucket);
  leaf->setParticles((Particle *)msg->data,msg->np);

  partReqs.decrRequests();
  partReqs.decrDeliveries(req.requestors.length());
  req.deliverParticles(msg->np);
}

void DataManager::recvNode(NodeReplyMsg *msg){
  map<Key,Request>::iterator it = nodeRequestTable.find(msg->key);
  CkAssert(it != nodeRequestTable.end());

  RRDEBUG("(%d) RECVD node REPLY for key %lu\n", 
          CkMyPe(), msg->key);

  Request &req = it->second;
  CkAssert(req.requestors.length() > 0);
  CkAssert(req.sent);
  CkAssert(req.msg == NULL);

  req.msg = msg;
  req.data = msg->data;
  
  // attach recvd subtree to appropriate point in local tree
  Node<ForceData> *node = req.parent;
  CkAssert(node != NULL);
  
  node->deserialize(msg->data, msg->nn);

  ostringstream oss;
  oss << "(" << CkMyPe() << ") key check: " << msg->key << endl;
  TreeChecker checker(oss);
  fillTrav.topDownTraversal_local(node,&checker);

  nodeReqs.decrRequests();
  nodeReqs.decrDeliveries(req.requestors.length());
  RRDEBUG("(%d) DELIVERING key %lu\n", 
          CkMyPe(), msg->key);

  req.deliverNode();
  RRDEBUG("(%d) DELIVERED key %lu\n", 
          CkMyPe(), msg->key);
}

#ifdef STATISTICS
void DataManager::traversalsDone(CmiUInt8 pnInter, CmiUInt8 ppInter, CmiUInt8 openCrit)
#else
void DataManager::traversalsDone()
#endif
{
  numTreePiecesDoneTraversals++;
#ifdef STATISTICS
  numInteractions[0] += pnInter;
  numInteractions[1] += ppInter;
  numInteractions[2] += openCrit;
#endif
  if(numTreePiecesDoneTraversals == numLocalTreePieces){
    finishIteration();
  }
}

void DataManager::finishIteration(){
  timers.stop(PHASE_TRAVERSAL);

#ifdef CHECK_TRAVERSAL_MASS
  checkTraversalMass();
#endif

  // can't advance particles here, because other PEs 
  // might not have finished their traversals yet, 
  // and therefore might need my particles

  CkAssert(nodeReqs.test());
  CkAssert(partReqs.test());

  InteractionChecker ic;
  fillTrav.postorderTraversal(root,&ic);

  DtReductionStruct dtred;
  findMinVByA(dtred);

#ifdef STATISTICS
  dtred.pnInteractions = numInteractions[0];
  dtred.ppInteractions = numInteractions[1];
  dtred.openCrit = numInteractions[2];
  numInteractions[0] = 0;
  numInteractions[1] = 0;
  numInteractions[2] = 0;
#endif

  CkCallback cb(CkIndex_DataManager::advance(NULL),thisProxy);
  contribute(sizeof(DtReductionStruct),&dtred,DtReductionType,cb);

}

void DataManager::advance(CkReductionMsg *msg){
  timers.start(PHASE_INTEGRATION);

  DtReductionStruct *dtred = (DtReductionStruct *)(msg->getData());
  if(dtred->haveNaN){
    CkPrintf("(%d) iteration %d NaN accel detected! Exit...\n", CkMyPe(), iteration);
    markNaNBuckets();
    printTree();
    CkCallback exitCb(CkCallback::ckExit);
    contribute(0,0,CkReduction::sum_int,exitCb);
    return;
  }

  // Before the integrator, not after: what is written is the state the
  // traversal just computed forces for, so positions, velocities,
  // accelerations and potentials in a frame all belong to t = iteration*dtime.
  if(globalParams.output.writeThisStep(iteration)){
    timers.start(PHASE_OUTPUT);
    writeSnapshot();
    timers.stop(PHASE_OUTPUT);
  }

  BoundingBox myBox;
  kickDriftKick(myBox.box,myBox.kineticEnergy,myBox.potentialEnergy);

  Real pad = 0.001;
  myBox.expand(pad);
  myBox.numParticles = myNumParticles;

  if(CkMyPe() == 0){
#ifdef STATISTICS
    CkPrintf("[STATS] node inter %lu part inter %lu open crit %lu dt %f\n", dtred->pnInteractions, dtred->ppInteractions, dtred->openCrit, globalParams.dtime);
#endif
  }

  CkAssert(pendingMoments.empty());
  // safe to reset here, since all tree pieces 
  // must have finished iteration
  freeCachedData();

  submittedParticles.length() = 0;
  haveRanges = false;
  myBuckets.length() = 0;
  doneTreeBuild = false;
  treeMomentsReady = false;
  numTreePiecesDoneTraversals = 0;

  firstSplitterRound = true;
  freeTree();
  nodeTable.clear();

  CkAssert(activeBins.getNumCounts() == 0);

  if(CkMyPe() == 0) delete[] keyRanges;
  else delete rangeMsg;
  keyRanges = NULL;

  timers.stop(PHASE_INTEGRATION);
  timers.finishStep();

  iteration++;
  CkCallback cb;
  if(iteration == globalParams.iterations){
    // The final step's energies would otherwise never be reported: nothing
    // calls decompose() again after this. Reduce them to Main, which prints
    // them and then exits.
    cb = CkCallback(CkIndex_Main::reportFinalEnergy(NULL),mainProxy);
    if (thisIndex == 0) 
      CkPrintf("(%d) finished all %d iterations with avg time %f\n", CkMyPe(), iteration, avgIterationRuntime/globalParams.iterations);
    contribute(sizeof(BoundingBox),&myBox,BoundingBoxGrowReductionType,cb);

    // The one time anything about timing is communicated: every PE's whole
    // table, reduced once, now that the run is over.
    if(timers.enabled()){
      const int n = timers.numValues();
      double *buf = new double[2*n];
      timers.fillReductionBuffer(buf);
      contribute(2*n*sizeof(double), buf, PhaseTimerReductionType,
                 CkCallback(CkIndex_Main::reportPhaseTimers(NULL),mainProxy));
      delete[] buf;
    }
  }
  else{
    cb = CkCallback(CkIndex_DataManager::recvUnivBoundingBox(NULL),thisProxy);
    contribute(sizeof(BoundingBox),&myBox,BoundingBoxGrowReductionType,cb);
  }
  delete msg;
}

void DataManager::recvUnivBoundingBox(CkReductionMsg *msg){
  BoundingBox &univBB = *((BoundingBox *)msg->getData());
  decompose(univBB);
  delete msg;
}

void DataManager::freeCachedData(){
  map<Key,Request>::iterator it;

  for(it = particleRequestTable.begin(); it != particleRequestTable.end(); it++){
    Request &request = it->second;
    CkAssert(request.sent);
    CkAssert(request.data != NULL);
    CkAssert(request.requestors.length() == 0);
    CkAssert(request.msg != NULL);
    
    // no need to set the particles of a cached
    // bucket to NULL: we will delete the bucket
    // anyway
    if(!request.parentCached){
      request.parent->setParticles(NULL,0);
    }

    delete (ParticleReplyMsg *)(request.msg);
  }

  for(it = nodeRequestTable.begin(); it != nodeRequestTable.end(); it++){
    Request &request = it->second;
    CkAssert(request.sent);
    CkAssert(request.data != NULL);
    CkAssert(request.requestors.length() == 0);
    CkAssert(request.msg != NULL);

    // tell the root of the nodes in this 
    // fetched entry that its children don't
    // exist anymore
    // cached parents may be deleted before
    // their children, so we don't set their
    // children
    if(!request.parentCached){
      request.parent->setChildren(NULL,0);
    }

    delete (NodeReplyMsg *)(request.msg);
  }

  nodeRequestTable.clear();
  particleRequestTable.clear();
}

void DataManager::quiescence(){
  CkPrintf("QUIESCENCE dm %d pieces done %d (%d) nodereq %d partreq %d\n",
              CkMyPe(),
              numTreePiecesDoneTraversals,
              numLocalTreePieces,
              nodeReqs.test(),
              partReqs.test()
              );
  
  CkCallback cb(CkIndex_Main::quiescenceExit(),mainProxy);
  contribute(0,0,CkReduction::sum_int,cb);
}

void DataManager::freeTree(){
  if(root != NULL){
    FreeTreeWorker<ForceData> freeWorker;
    fillTrav.postorderTraversal(root,&freeWorker); 
    delete root;
    root = NULL;
  }
}

void DataManager::printTree(){
  ostringstream oss;
  oss << "pe" << CkMyPe() << "." << iteration << ".dot";

  ofstream ofs(oss.str().c_str());

  ofs << "digraph PE" << CkMyPe() << "_" << iteration << " {" << endl;
  ofs << "node [style=\"filled\"]" << endl;
  if(root != NULL){
    Node<ForceData> &rootRef = *root;
    ofs << rootRef;
  }
  ofs << "}" << endl;
  ofs.close();
}

void DataManager::addBucketNodeInteractions(Key k, CmiUInt8 pn){
#ifdef CHECK_NUM_INTERACTIONS
  Node<ForceData> *node = nodeTable[k];
  node->addNodeInteractions(pn);
#endif
}

void DataManager::addBucketPartInteractions(Key k, CmiUInt8 pp){
#ifdef CHECK_NUM_INTERACTIONS
  Node<ForceData> *node = nodeTable[k];
  node->addPartInteractions(pp);
#endif
}

#ifdef CHECK_TRAVERSAL_MASS
// See the note on traversalMass in Worker.h. Reports the spread rather than
// comparing against a known total, because a PE does not hold one: if the
// traversal is exact every bucket everywhere lands on the same figure, so any
// spread at all is the bug.
void DataManager::checkTraversalMass(){
  std::map<Key,Real> &seen = traversalMass[CkMyPe()];
  if(seen.empty()) return;

  Real lo = seen.begin()->second;
  Real hi = lo;
  Key loKey = seen.begin()->first;
  Key hiKey = loKey;
  for(std::map<Key,Real>::iterator it = seen.begin(); it != seen.end(); ++it){
    if(it->second < lo){ lo = it->second; loKey = it->first; }
    if(it->second > hi){ hi = it->second; hiKey = it->first; }
  }

  CkPrintf("[MASS] step %d pe %d buckets %d interacted mass [%.12g, %.12g] spread %.3g"
           " (min at key %llu, max at key %llu)\n",
           iteration, CkMyPe(), (int)seen.size(), lo, hi, (double)(hi-lo),
           (unsigned long long)loKey, (unsigned long long)hiKey);

  seen.clear();
}
#endif

// One ParaView snapshot of this PE's particles. Every PE writes its own piece
// with no communication and no gather; PE 0 additionally writes the two small
// index files that tie the pieces together. The piece names are derived from
// the step and PE numbers, so PE 0 can name them all without hearing from
// anyone -- it may well write the index before the pieces themselves exist,
// which is harmless, since nothing reads these until the run is over.
void DataManager::writeSnapshot(){
  const OutputConfig &out = globalParams.output;
  const std::string stem = vtkStepStem(out.prefix, iteration);

  std::string err;
  if(!writeVtkPiece(vtkPieceName(stem, CkMyPe()), myParticles.getVec(),
                    myNumParticles, CkMyPe(), err)){
    // A missing directory is the overwhelmingly likely cause, and silently
    // producing an unopenable dataset after a long run is worse than stopping.
    CkPrintf("(%d) output: %s\n", CkMyPe(), err.c_str());
    CkAbort("cannot write ParaView output\n");
  }

  if(CkMyPe() != 0) return;

  writtenSteps.push_back(iteration);

  // The collection is rewritten, not appended to, so that a run stopped part
  // way through still leaves a .pvd naming exactly the frames that exist.
  if(!writeVtkParallelIndex(stem + ".pvtp", stem, CkNumPes(), err) ||
     !writeVtkCollection(out.prefix + ".pvd", out.prefix, writtenSteps,
                         globalParams.dtime, err)){
    CkPrintf("(%d) output: %s\n", CkMyPe(), err.c_str());
    CkAbort("cannot write ParaView output\n");
  }
}

void DataManager::kickDriftKick(OrientedBox<Real> &box, Real &kineticEnergy, Real &potentialEnergy){
  Particle *pstart = myParticles.getVec();
  Particle *pend = myParticles.getVec()+myNumParticles;
  Real dt_k1, dt_k2;

  // The traversal kernel accumulates potential and acceleration with G factored
  // out (i.e. as though G == 1) so that the inner loop stays a plain sum. G is
  // restored here, once per particle rather than once per interaction.
  const Real G = globalParams.G;

  if(iteration == 0){
    dt_k1 = globalParams.dtime;
    // won't have KE stored by a previous 
    // iteration for this one
    for(Particle *p = pstart; p != pend; p++){
      kineticEnergy += p->mass*(p->velocity.lengthSquared()); 
    }
    kineticEnergy /= 2.0;
  }
  else{
    dt_k1 = globalParams.dthf;
    kineticEnergy = savedEnergy;
    savedEnergy = 0.0;
  }

  dt_k2 = globalParams.dthf;

  // A bucket is always opened against itself (its own centre of mass sits at
  // zero distance), so every particle picks up one particle-particle term
  // against itself: -m_i/sqrt(0 + eps^2) = -m_i/eps. Softening keeps that
  // finite rather than infinite, so it silently biases E_P by -sum_i m_i^2/eps
  // instead of blowing up. Whitepaper Eq. 12 sums over i<j only, so it is
  // removed here. The acceleration needs no such correction: its self term is
  // mor3*dr with dr == 0, which is exactly zero.
  const Real eps = sqrt(globalParams.epssq);

  for(Particle *p = pstart; p != pend; p++){
    const Real selfPotential = (eps > 0.0) ? -p->mass/eps : 0.0;
    const Real phi = G*(p->potential - selfPotential);
    const Vector3D<Real> accel = G*p->acceleration;

    // phi_i = -sum_{j != i} m_j / r_ij, so sum_i m_i * phi_i visits every pair
    // twice. Whitepaper Eq. 12 defines E_P = -G * sum_{i<j} m_i m_j / r_ij,
    // over unordered pairs, hence the one half.
    potentialEnergy += 0.5*p->mass*phi;

    // kick
    p->velocity += dt_k1*accel;
    savedEnergy += p->mass*(p->velocity.lengthSquared()); 
    // drift
    p->position += globalParams.dtime*p->velocity;
    // kick
    p->velocity += dt_k2*accel;
    
    box.grow(p->position);

    p->acceleration = Vector3D<Real>(0.0);
    p->potential = 0.0;

  }
  savedEnergy /= 2.0;
}

void DataManager::findMinVByA(DtReductionStruct &dtred){
  if(myNumParticles == 0) {
    dtred.haveNaN = false;
    dtred.vbya = -1.0; 
    return;
  }
  
  dtred.haveNaN = false;

  for(int i = 0; i < myNumParticles; i++){
    Real v = myParticles[i].velocity.length();
    Real a = myParticles[i].acceleration.length();
    CkAssert(!isnan(v));
    if(isnan(a)) dtred.haveNaN = true;
   }

  dtred.vbya = -1.0;
}

void DataManager::markNaNBuckets(){
  for(int i = 0; i < myBuckets.length(); i++){
    Node<ForceData> *bucket = myBuckets[i];
    Particle *part = bucket->getParticles();
    int numParticles = bucket->getNumParticles();
    for(int j = 0; j < numParticles; j++){
      if(isnan(part[j].acceleration.length())){
        bucket->setType(Invalid);
        break;
      }
    }
  }
}



#include "Traversal_defs.h"

