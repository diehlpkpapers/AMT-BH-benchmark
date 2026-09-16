#ifndef __DESCRIPTOR_H__
#define __DESCRIPTOR_H__

#include "defines.h"
#include "OrientedBox.h"
#include "MultipoleMoments.h"
#include "charm++.h"

struct BoundingBox {
  OrientedBox<Real> box;
  int numParticles;
  // Tracked separately so that the whitepaper's E_K and E_P (and their sum
  // E_T, which should stay constant) can each be reported.
  Real kineticEnergy;
  Real potentialEnergy;

  BoundingBox(){
    reset();
  }

  void reset(){
    numParticles = 0;
    box.reset();
    kineticEnergy = 0.0;
    potentialEnergy = 0.0;
  }

  Real totalEnergy() const { return kineticEnergy + potentialEnergy; }

  void grow(const Vector3D<Real> &v){
    box.grow(v);
  }

  void grow(const BoundingBox &other){
    if(other.numParticles == 0) return;
    if(numParticles == 0){
      *this = other;
    }
    else{
      box.grow(other.box);
      numParticles += other.numParticles;
      kineticEnergy += other.kineticEnergy;
      potentialEnergy += other.potentialEnergy;
    }
  }

  // Grow the box outward by `pad` times its own extent in each dimension.
  //
  // NOTE: the original scaled each corner by (1 +/- pad). For a corner with a
  // negative coordinate that moves the corner *inward*, shrinking the box, so
  // particles on the boundary could end up outside the universe and hash to
  // out-of-range Morton keys. Padding by a fraction of the size is symmetric
  // and cannot shrink the box.
  void expand(Real pad){
    Vector3D<Real> delta = (box.greater_corner - box.lesser_corner)*pad;
    // A degenerate axis (all particles coplanar, or a single particle) would
    // leave a zero-width box and divide by zero when hashing coordinates.
    if(delta.x <= 0.0) delta.x = pad;
    if(delta.y <= 0.0) delta.y = pad;
    if(delta.z <= 0.0) delta.z = pad;
    box.lesser_corner -= delta;
    box.greater_corner += delta;
  }

  void pup(PUP::er &p){
    p | box;
    p | numParticles;
    p | kineticEnergy;
    p | potentialEnergy;
  }

};

// how many particles do I have under this node?
// what are the largest and smallest keys among them?
struct NodeDescriptor {
  int numParticles;
  Key nodeKey;

  Key smallestKey;
  Key largestKey;

  NodeDescriptor() {
    numParticles = 0;
    largestKey = Key(0);
    smallestKey = ~largestKey;
    nodeKey = smallestKey;
  }

  NodeDescriptor(int np, Key nk, Key sk, Key lk) : 
    numParticles(np), 
    nodeKey(nk), smallestKey(sk), largestKey(lk) 
  {
  }

 void grow(const NodeDescriptor &other){
   if(other.numParticles == 0){
      return;
    }

    if(numParticles == 0){
      smallestKey = other.smallestKey;
      largestKey = other.largestKey;
      numParticles = other.numParticles;
    } else {
      if(smallestKey > other.smallestKey) smallestKey = other.smallestKey;
      if(largestKey < other.largestKey) largestKey = other.largestKey;
      numParticles += other.numParticles;
    }

  }
};

class TreePiece;
struct ParticleMsg;
struct TreePieceDescriptor {
  CkVec<ParticleMsg*> *vec;
  TreePiece *owner;
  int index;
  int numParticles;
  Key smallestKey;
  Key largestKey;

  int bucketStartIdx;
  int bucketEndIdx;

  TreePieceDescriptor() : 
    vec(NULL), owner(NULL), numParticles(0), index(-1), smallestKey(~Key(0)), largestKey(Key(0))
  {
  }

  TreePieceDescriptor(CkVec<ParticleMsg*> *v, int np, TreePiece *o, int i, Key sk, Key lk) : 
    vec(v), owner(o), index(i), smallestKey(sk), largestKey(lk), numParticles(np)
  {
  }

  TreePieceDescriptor(int i) : 
    vec(NULL), owner(NULL), numParticles(0), index(i), smallestKey(~Key(0)), largestKey(Key(0))
  {
  }

  bool operator<=(const TreePieceDescriptor &t){
    return smallestKey <= t.smallestKey;
  }
  bool operator>=(const TreePieceDescriptor &t){
    return smallestKey >= t.smallestKey;
  }
};

struct ForceData {
  OrientedBox<Real> box;
  MultipoleMoments moments;

  ForceData() 
  {
  }
};

struct DtReductionStruct {
#ifdef STATISTICS 
  CmiUInt8 pnInteractions;
  CmiUInt8 ppInteractions;
  CmiUInt8 openCrit;
#endif
  Real vbya;
  bool haveNaN;

  DtReductionStruct &operator+=(const DtReductionStruct &other){
#ifdef STATISTICS
    pnInteractions += other.pnInteractions;
    ppInteractions += other.ppInteractions;
    openCrit += other.openCrit;
#endif
    if(other.haveNaN) haveNaN = true;

    if(other.vbya < 0.0) {} 
    else if(vbya < 0.0) vbya = other.vbya;
    else if(other.vbya < vbya) vbya = other.vbya;

    return *this;
  }
};


#endif
