#pragma once

#include "particle.h"

#include <string>
#include <vector>

namespace bh {

// Reads the whitepaper's headerless input format:
// id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z
std::vector<Particle> read_csv(const std::string& path);

// Writes a legacy VTK ASCII PolyData snapshot (points + mass scalar +
// velocity vector), readable directly by Paraview.
void write_vtk(const std::string& path, const std::vector<Particle>& particles);

}  // namespace bh
