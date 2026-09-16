#ifndef __WORKER_H__
#define __WORKER_H__

#include "Node.h"
#include "util.h"
#include "MultipoleMoments.h"
#include "Descriptor.h"

#include <map>
using namespace std;

// worker abstract class
template<typename T>
class CutoffWorker {
  public:
  virtual int work(Node<T> *) = 0; 
  virtual void work(ExternalParticle *) {}
  virtual void bucketDone(Key k) {}
  virtual void *getContext() {return NULL;}
  virtual void setContext(void *context) {}
  virtual void done() {}
};

class DataManager;
class ParticleFlushWorker : public CutoffWorker<NodeDescriptor> {
  int leafCnt;
  DataManager *dataManager;

  public:
  ParticleFlushWorker(DataManager *dm) : 
    dataManager(dm),
    leafCnt(0)
  {
  }

  int work(Node<NodeDescriptor> *node);
  int getNumLeaves(){
    return leafCnt;
  }
};

class MomentsWorker : public CutoffWorker<ForceData> {

  // tree pieces on this PE, in sorted order
  // this allows us to mark the nodes that are
  // internal to each PE. The last PE in this list
  // is a dummy value, and is equal to the number of
  // useful treepieces+1
  CkVec<TreePieceDescriptor> &peTreePieces;
  map<Key,Node<ForceData>*> &nodeTable;
  CkVec<Node<ForceData>*> &buckets;

  int curTP;

  public: 
  MomentsWorker(CkVec<TreePieceDescriptor> &petps,
                map<Key,Node<ForceData>*> &tab,
                CkVec<Node<ForceData>*> &bucks
                ) : 
    peTreePieces(petps),
    nodeTable(tab),
    buckets(bucks),
    curTP(0)
  {
  }

  int work(Node<ForceData> *node);
  void setLeafType(Node<ForceData> *leaf);
  void setTypeFromChildren(Node<ForceData> *node);
};

#ifdef CHECK_TRAVERSAL_MASS
#include "common.h"
// Every bucket must end a step having interacted with the entire system, once:
// the mass it sees directly plus the mass of every node it approximated should
// come to the total mass, identically for every bucket on every PE. It is the
// invariant that catches an interaction being dropped or applied twice, which
// energies alone only hint at. Keyed by bucket, per PE.
extern std::map<Key,Real> traversalMass[];
#define MAX_TRAVERSAL_MASS_PES 1024
#endif

class State;
class TraversalWorker : public CutoffWorker<ForceData> {
  protected:
  TreePiece *ownerTreePiece;
  State *state;
  
  Node<ForceData> *currentBucket;

  TraversalWorker() : 
    ownerTreePiece(NULL),
    currentBucket(NULL)
  {
  }

  public:

  void reset(TreePiece *owner, State *s, Node<ForceData> *bucket){
    ownerTreePiece = owner;
    state = s;
    currentBucket = bucket;
  }

  Node<ForceData> *getCurrentBucket(){
    return currentBucket;
  }

  void *getContext(){
    return (void *) currentBucket;
  }

  void setContext(void *context){
    currentBucket = (Node<ForceData> *) context;
  }

  int work(Node<ForceData> *node);
  void work(ExternalParticle *particle);
  void bucketDone(Key k);
  
  virtual void done() {}
  virtual bool getKeep(NodeType type) = 0;

  // Both traversals keep Boundary nodes -- a Boundary node is the shared spine
  // between the local and the remote part of the tree, so each has to be able
  // to walk down through it. When such a node is *not* opened, though, its
  // multipole stands for its whole subtree, local and remote alike, and only
  // one of the two may apply it. The local traversal does; see
  // TraversalWorker::work.
  virtual bool computesUnopenedBoundary() const = 0;
};

class LocalTraversalWorker : public TraversalWorker {
  static const bool keep[];
  public:
  LocalTraversalWorker() : TraversalWorker() {}
  bool getKeep(NodeType type);
  bool computesUnopenedBoundary() const { return true; }
};

class RemoteTraversalWorker : public TraversalWorker {
  static const bool keep[];
  public:
  RemoteTraversalWorker() : TraversalWorker() {}
  void done();
  bool getKeep(NodeType type);
  bool computesUnopenedBoundary() const { return false; }
};

class TreeSizeWorker : public CutoffWorker<ForceData> {

  int numNodes;
  int cutoffDepth;

  public:
  TreeSizeWorker(int d) : 
    cutoffDepth(d),
    numNodes(0)
  {
  }

  int work(Node<ForceData> *node);
  int getNumNodes(){
    return numNodes;
  }
};

template<typename T>
class FreeTreeWorker : public CutoffWorker<T> {
  public:
  int work(Node<T> *node){
    if(node->getNumChildren() > 0){
      /*
         CkPrintf("(%d) deleting %d children of %lu ptr %lx\n", 
         CkMyPe(), 
         node->getNumChildren(),
         node->getKey(),
         node->getChildren()
         );
         */
      delete[] node->getChildren();
    }
    return 1;
  }
};

class TreeChecker : public CutoffWorker<ForceData> {
  ostream &os;
  public: 
  TreeChecker(ostream &o) : os(o) {}
  int work(Node<ForceData> *node);
};

class InteractionChecker : public CutoffWorker<ForceData> {
  public:
  InteractionChecker() {}
  int work(Node<ForceData> *node);
};

#endif
