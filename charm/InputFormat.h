#ifndef __INPUT_FORMAT_H__
#define __INPUT_FORMAT_H__

#include "common.h"

#include <cstdlib>
#include <cctype>
#include <string>

// Which on-disk layout the input file uses. Select it with -format=, or let
// it be inferred from the file's extension.
//
//   INPUT_BINARY  the format the bundled ./plummer and ./gen write: two ints
//                 (nbody, ndims), one Real (tnow), then REALS_PER_PARTICLE
//                 Reals per body -- x y z vx vy vz mass soft. The Real width
//                 follows this build, so a file is only readable by a build
//                 with a matching sizeof(Real).
//
//   INPUT_CSV     the benchmark's portable format (spec v1.0 sec. 5), written
//                 by https://github.com/vancraar/DataGenerator:
//                     id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z
//                 Text, so it is the format that can be shared across the
//                 runtimes being compared.
enum InputFormat { INPUT_BINARY = 0, INPUT_CSV = 1 };

// The seven quantities a row has to supply. `id` is deliberately not among
// them: Particle has nowhere to put it, and the ordering the simulation cares
// about is the Morton key, not the generator's numbering.
enum CsvField {
  CSV_MASS = 0,
  CSV_POS_X, CSV_POS_Y, CSV_POS_Z,
  CSV_VEL_X, CSV_VEL_Y, CSV_VEL_Z,
  CSV_NUM_FIELDS
};

// Where each field sits in a row, and where the rows start.
//
// Columns are located by name from the header rather than assumed positional,
// because the spec document and the generator disagree on spelling (posx vs.
// pos_x). A file with no header falls back to the spec's column order.
struct CsvLayout {
  int column[CSV_NUM_FIELDS];  // field -> zero-based column index
  int numColumns;              // columns the header declared
  long dataOffset;             // byte offset of the first data row

  CsvLayout() : numColumns(0), dataOffset(0) {
    setSpecOrder();
  }

  // id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z
  void setSpecOrder(){
    column[CSV_MASS]  = 1;
    column[CSV_POS_X] = 2;
    column[CSV_POS_Y] = 3;
    column[CSV_POS_Z] = 4;
    column[CSV_VEL_X] = 5;
    column[CSV_VEL_Y] = 6;
    column[CSV_VEL_Z] = 7;
    numColumns = 8;
  }

  // Lower-case a column name and drop quotes, spaces and underscores, so that
  // pos_x, "posx" and POS X all reduce to the same token.
  static std::string canonicalize(const std::string &name){
    std::string out;
    for(size_t i = 0; i < name.size(); i++){
      char c = name[i];
      if(c == '_' || c == '"' || c == '\'' || c == '\r' ||
         isspace((unsigned char)c)) continue;
      out += (char)tolower((unsigned char)c);
    }
    return out;
  }

  static int fieldFromName(const std::string &canon){
    if(canon == "mass" || canon == "m") return CSV_MASS;
    if(canon == "posx" || canon == "x") return CSV_POS_X;
    if(canon == "posy" || canon == "y") return CSV_POS_Y;
    if(canon == "posz" || canon == "z") return CSV_POS_Z;
    if(canon == "velx" || canon == "vx") return CSV_VEL_X;
    if(canon == "vely" || canon == "vy") return CSV_VEL_Y;
    if(canon == "velz" || canon == "vz") return CSV_VEL_Z;
    return -1;
  }

  // True if the line looks like column names rather than numbers: a data row
  // begins with a parseable number, a header does not.
  static bool looksLikeHeader(const std::string &line){
    const char *start = line.c_str();
    char *end = NULL;
    strtod(start, &end);
    return end == start;
  }

  // Fills column[] from a header line. Returns false, leaving a diagnostic in
  // `missing`, if any required field is absent.
  bool parseHeader(const std::string &line, std::string &missing){
    for(int f = 0; f < CSV_NUM_FIELDS; f++) column[f] = -1;

    int index = 0;
    size_t pos = 0;
    while(true){
      size_t comma = line.find(',', pos);
      size_t len = (comma == std::string::npos ? line.size() : comma) - pos;
      int field = fieldFromName(canonicalize(line.substr(pos, len)));
      if(field >= 0 && column[field] < 0) column[field] = index;
      index++;
      if(comma == std::string::npos) break;
      pos = comma + 1;
    }
    numColumns = index;

    static const char *names[CSV_NUM_FIELDS] =
        { "mass", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z" };
    for(int f = 0; f < CSV_NUM_FIELDS; f++){
      if(column[f] < 0){
        missing = names[f];
        return false;
      }
    }
    return true;
  }

  // Pulls the seven mapped columns out of one row. Walks the row once and
  // parses in place -- strtod stops at the comma on its own, so no substring
  // is copied. Returns false if a mapped column is missing or unparseable.
  bool parseRow(const std::string &line, Real out[CSV_NUM_FIELDS]) const {
    int filled = 0;
    int index = 0;
    size_t pos = 0;
    while(pos <= line.size()){
      for(int f = 0; f < CSV_NUM_FIELDS; f++){
        if(column[f] != index) continue;
        const char *fieldStart = line.c_str() + pos;
        char *end = NULL;
        double value = strtod(fieldStart, &end);
        if(end == fieldStart) return false;
        out[f] = (Real)value;
        filled++;
      }
      size_t comma = line.find(',', pos);
      if(comma == std::string::npos) break;
      pos = comma + 1;
      index++;
    }
    return filled == CSV_NUM_FIELDS;
  }
};

// A record is a line with something other than line terminators on it, so that
// stray blank lines (a trailing newline, most often) are not counted as
// particles. The scan on PE 0 and the readers on every PE must agree on this,
// or the per-PE record ranges will not line up with the total.
inline bool isBlankCsvLine(const std::string &line){
  for(size_t i = 0; i < line.size(); i++){
    if(line[i] != '\r' && line[i] != '\n') return false;
  }
  return true;
}

#ifdef __CHARMC__
#include "pup.h"

inline void operator|(PUP::er &p, CsvLayout &c){
  for(int f = 0; f < CSV_NUM_FIELDS; f++) p | c.column[f];
  p | c.numColumns;
  p | c.dataOffset;
}
#endif // __CHARMC__

#endif // __INPUT_FORMAT_H__
