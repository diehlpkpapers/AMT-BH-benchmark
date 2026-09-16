#ifndef __VTK_OUTPUT_H__
#define __VTK_OUTPUT_H__

#include "common.h"
#include "Particle.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Particle snapshots for ParaView, spec v1.0 section 6.
//
// The output is VTK's XML PolyData family, which is what ParaView opens
// natively with no import step:
//
//   <prefix>_<step>_pe<n>.vtp   one file per PE per snapshot: that PE's
//                               particles, written by that PE. Nothing is
//                               gathered, so the write costs one local file
//                               per PE and no communication.
//   <prefix>_<step>.pvtp        names the pieces that make up one snapshot,
//                               so ParaView loads them as a single dataset.
//   <prefix>.pvd                names the snapshots and their simulation
//                               times, so ParaView animates them.
//
// Open the .pvd. The other two levels exist for it.
//
// Point data travels as raw little-endian doubles in the appended-data
// section rather than as text: a text .vtp is roughly three times the size
// and much slower to write, and the output phase is timed. The XML header
// declares a byte offset per array, so the offsets have to be known before
// any data is written -- hence the fixed array table below, from which both
// the offsets and the blocks are generated.
//
// The particles a snapshot holds are the state the forces were computed
// from: positions, velocities, accelerations and potentials all belong to
// the same instant, t = step*dtime. That is why the write happens before the
// integrator runs, and why there is no snapshot after the final step -- the
// final positions exist, but their accelerations and potentials do not, and
// a frame whose fields are half valid is worse than no frame.

struct OutputConfig {
  std::string prefix;   // empty: no output at all
  int frequency;        // write every frequency-th step

  OutputConfig() : frequency(1) {}

  bool enabled() const { return !prefix.empty(); }
  bool writeThisStep(int step) const {
    return enabled() && frequency > 0 && (step % frequency) == 0;
  }
};

// The arrays a .vtp piece carries, in the order they are appended.
//
// The first three are geometry and topology. VTK will render a point cloud
// with no cells at all, but only in ParaView's Points representation -- the
// vertex cells are what make the particles visible in the Surface
// representation the reader lands in by default, and they are what the Glyph
// filter attaches to. They cost 8 bytes a particle, which is worth it.
//
// The rest are point data, i.e. the per-particle fields ParaView colours by.
// `pe` is not physics: it is which PE owned the particle at that step, which
// is what makes the decomposition visible.
enum VtkArrayId {
  VTK_POINTS = 0,
  VTK_CONNECTIVITY,
  VTK_OFFSETS,
  VTK_MASS,            // first point-data array
  VTK_VELOCITY,
  VTK_ACCELERATION,
  VTK_POTENTIAL,
  VTK_PE,
  VTK_KEY,             // last point-data array
  VTK_NUM_ARRAYS
};

#define VTK_FIRST_POINT_ARRAY VTK_MASS
#define VTK_LAST_POINT_ARRAY  VTK_KEY

struct VtkArrayDesc {
  const char *name;
  const char *type;    // VTK's name for the element type
  int numComponents;
  int width;           // bytes per component
};

inline const VtkArrayDesc &vtkArrayDesc(int id){
  static const VtkArrayDesc descs[VTK_NUM_ARRAYS] = {
    { "Points",       "Float64", 3, 8 },
    { "connectivity", "Int32",   1, 4 },
    { "offsets",      "Int32",   1, 4 },
    { "mass",         "Float64", 1, 8 },
    { "velocity",     "Float64", 3, 8 },
    { "acceleration", "Float64", 3, 8 },
    { "potential",    "Float64", 1, 8 },
    { "pe",           "Int32",   1, 4 },
    { "key",          "UInt64",  1, 8 }
  };
  return descs[id];
}

inline size_t vtkArrayBytes(int id, int numParticles){
  const VtkArrayDesc &d = vtkArrayDesc(id);
  return (size_t)numParticles*d.numComponents*d.width;
}

// VTK records the byte order the writer used rather than mandating one, so
// report this machine's honestly instead of assuming little-endian.
inline const char *vtkByteOrder(){
  const uint32_t one = 1;
  return (*(const char *)&one) ? "LittleEndian" : "BigEndian";
}

// Fills `buf` with one array's raw bytes. The Real width of the simulation
// does not reach the file: everything is written as the type the header
// declares, so a snapshot is readable whatever this build was compiled with.
inline void fillVtkBlock(int id, const Particle *p, int n, int pe,
                         std::vector<char> &buf){
  const size_t bytes = vtkArrayBytes(id, n);
  buf.resize(bytes);
  if(bytes == 0) return;

  switch(id){
    case VTK_POINTS: {
      double *o = (double *)&buf[0];
      for(int i = 0; i < n; i++){
        o[3*i+0] = (double)p[i].position.x;
        o[3*i+1] = (double)p[i].position.y;
        o[3*i+2] = (double)p[i].position.z;
      }
      break;
    }
    case VTK_CONNECTIVITY: {
      int32_t *o = (int32_t *)&buf[0];
      for(int i = 0; i < n; i++) o[i] = (int32_t)i;
      break;
    }
    case VTK_OFFSETS: {
      int32_t *o = (int32_t *)&buf[0];
      for(int i = 0; i < n; i++) o[i] = (int32_t)(i+1);
      break;
    }
    case VTK_MASS: {
      double *o = (double *)&buf[0];
      for(int i = 0; i < n; i++) o[i] = (double)p[i].mass;
      break;
    }
    case VTK_VELOCITY: {
      double *o = (double *)&buf[0];
      for(int i = 0; i < n; i++){
        o[3*i+0] = (double)p[i].velocity.x;
        o[3*i+1] = (double)p[i].velocity.y;
        o[3*i+2] = (double)p[i].velocity.z;
      }
      break;
    }
    case VTK_ACCELERATION: {
      double *o = (double *)&buf[0];
      for(int i = 0; i < n; i++){
        o[3*i+0] = (double)p[i].acceleration.x;
        o[3*i+1] = (double)p[i].acceleration.y;
        o[3*i+2] = (double)p[i].acceleration.z;
      }
      break;
    }
    case VTK_POTENTIAL: {
      double *o = (double *)&buf[0];
      for(int i = 0; i < n; i++) o[i] = (double)p[i].potential;
      break;
    }
    case VTK_PE: {
      int32_t *o = (int32_t *)&buf[0];
      for(int i = 0; i < n; i++) o[i] = (int32_t)pe;
      break;
    }
    case VTK_KEY: {
      uint64_t *o = (uint64_t *)&buf[0];
      for(int i = 0; i < n; i++) o[i] = (uint64_t)p[i].key;
      break;
    }
  }
}

// <prefix>_00007, the stem every file belonging to step 7 is built from.
inline std::string vtkStepStem(const std::string &prefix, int step){
  char suffix[16];
  snprintf(suffix, sizeof(suffix), "_%05d", step);
  return prefix + suffix;
}

inline std::string vtkPieceName(const std::string &stem, int pe){
  char suffix[24];
  snprintf(suffix, sizeof(suffix), "_pe%d.vtp", pe);
  return stem + suffix;
}

// A .pvtp and a .pvd name their members relative to their own directory, so
// what goes inside them is the file name alone, not the path used to open it.
inline std::string vtkBaseName(const std::string &path){
  size_t slash = path.find_last_of('/');
  return (slash == std::string::npos) ? path : path.substr(slash+1);
}

// One PE's particles for one step. Returns false on any I/O failure, with a
// reason in `err`; the caller decides whether that is fatal.
inline bool writeVtkPiece(const std::string &path, const Particle *p, int n,
                          int pe, std::string &err){
  std::ofstream out(path.c_str(), std::ios::out | std::ios::binary);
  if(!out.is_open()){
    err = "cannot open " + path + " for writing";
    return false;
  }

  // Each appended block is a UInt64 byte count followed by that many bytes,
  // so an array's offset is the sum of the sizes of everything before it.
  size_t offset[VTK_NUM_ARRAYS];
  size_t running = 0;
  for(int id = 0; id < VTK_NUM_ARRAYS; id++){
    offset[id] = running;
    running += sizeof(uint64_t) + vtkArrayBytes(id, n);
  }

  std::ostringstream x;
  x << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\""
    << vtkByteOrder() << "\" header_type=\"UInt64\">\n"
    << "  <PolyData>\n"
    << "    <Piece NumberOfPoints=\"" << n << "\" NumberOfVerts=\"" << n
    << "\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";

  for(int id = 0; id < VTK_NUM_ARRAYS; id++){
    const VtkArrayDesc &d = vtkArrayDesc(id);
    if(id == VTK_POINTS) x << "      <Points>\n";
    if(id == VTK_CONNECTIVITY) x << "      <Verts>\n";
    if(id == VTK_FIRST_POINT_ARRAY)
      x << "      <PointData Scalars=\"potential\" Vectors=\"velocity\">\n";

    x << "        <DataArray type=\"" << d.type << "\" Name=\"" << d.name
      << "\" NumberOfComponents=\"" << d.numComponents
      << "\" format=\"appended\" offset=\"" << offset[id] << "\"/>\n";

    if(id == VTK_POINTS) x << "      </Points>\n";
    if(id == VTK_OFFSETS) x << "      </Verts>\n";
    if(id == VTK_LAST_POINT_ARRAY) x << "      </PointData>\n";
  }

  x << "    </Piece>\n"
    << "  </PolyData>\n"
    << "  <AppendedData encoding=\"raw\">\n"
    << "   _";

  const std::string header = x.str();
  out.write(header.data(), header.size());

  std::vector<char> buf;
  for(int id = 0; id < VTK_NUM_ARRAYS; id++){
    const uint64_t bytes = (uint64_t)vtkArrayBytes(id, n);
    out.write((const char *)&bytes, sizeof(bytes));
    if(bytes == 0) continue;
    fillVtkBlock(id, p, n, pe, buf);
    out.write(&buf[0], (std::streamsize)bytes);
  }

  const char *trailer = "\n  </AppendedData>\n</VTKFile>\n";
  out.write(trailer, strlen(trailer));

  out.close();
  if(out.fail()){
    err = "write failed on " + path;
    return false;
  }
  return true;
}

// The snapshot's index: which pieces belong to it and what fields they carry.
// Only point data is declared here -- the pieces' own headers describe their
// geometry and topology.
inline bool writeVtkParallelIndex(const std::string &path,
                                  const std::string &stem, int numPieces,
                                  std::string &err){
  std::ofstream out(path.c_str());
  if(!out.is_open()){
    err = "cannot open " + path + " for writing";
    return false;
  }

  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"PPolyData\" version=\"1.0\" byte_order=\""
      << vtkByteOrder() << "\">\n"
      << "  <PPolyData GhostLevel=\"0\">\n"
      << "    <PPointData Scalars=\"potential\" Vectors=\"velocity\">\n";

  for(int id = VTK_FIRST_POINT_ARRAY; id <= VTK_LAST_POINT_ARRAY; id++){
    const VtkArrayDesc &d = vtkArrayDesc(id);
    out << "      <PDataArray type=\"" << d.type << "\" Name=\"" << d.name
        << "\" NumberOfComponents=\"" << d.numComponents << "\"/>\n";
  }

  out << "    </PPointData>\n"
      << "    <PPoints>\n"
      << "      <PDataArray type=\"" << vtkArrayDesc(VTK_POINTS).type
      << "\" Name=\"Points\" NumberOfComponents=\"3\"/>\n"
      << "    </PPoints>\n";

  const std::string base = vtkBaseName(stem);
  for(int pe = 0; pe < numPieces; pe++){
    out << "    <Piece Source=\"" << vtkPieceName(base, pe) << "\"/>\n";
  }

  out << "  </PPolyData>\n</VTKFile>\n";
  out.close();
  if(out.fail()){
    err = "write failed on " + path;
    return false;
  }
  return true;
}

// The animation: every snapshot so far, with the simulation time it holds.
// Rewritten after each snapshot rather than once at the end, so a run that is
// cut short still leaves something ParaView can open.
inline bool writeVtkCollection(const std::string &path,
                               const std::string &prefix,
                               const std::vector<int> &steps, Real dtime,
                               std::string &err){
  std::ofstream out(path.c_str());
  if(!out.is_open()){
    err = "cannot open " + path + " for writing";
    return false;
  }

  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\""
      << vtkByteOrder() << "\">\n"
      << "  <Collection>\n";

  out.precision(17);
  const std::string base = vtkBaseName(prefix);
  for(size_t i = 0; i < steps.size(); i++){
    out << "    <DataSet timestep=\"" << (double)steps[i]*dtime
        << "\" group=\"\" part=\"0\" file=\""
        << vtkStepStem(base, steps[i]) << ".pvtp\"/>\n";
  }

  out << "  </Collection>\n</VTKFile>\n";
  out.close();
  if(out.fail()){
    err = "write failed on " + path;
    return false;
  }
  return true;
}

#ifdef __CHARMC__
#include "pup.h"
#include "pup_stl.h"

inline void operator|(PUP::er &p, OutputConfig &c){
  p | c.prefix;
  p | c.frequency;
}
#endif // __CHARMC__

#endif // __VTK_OUTPUT_H__
