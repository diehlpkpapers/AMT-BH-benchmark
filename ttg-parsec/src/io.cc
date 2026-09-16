#include "io.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace bh {

std::vector<Particle> read_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open input file: " + path);

  std::vector<Particle> particles;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string field;
    Particle p;
    std::getline(ss, field, ',');
    p.id = static_cast<ParticleId>(std::stoul(field));
    std::getline(ss, field, ',');
    p.mass = std::stod(field);
    std::getline(ss, field, ',');
    p.pos.x = std::stod(field);
    std::getline(ss, field, ',');
    p.pos.y = std::stod(field);
    std::getline(ss, field, ',');
    p.pos.z = std::stod(field);
    std::getline(ss, field, ',');
    p.vel.x = std::stod(field);
    std::getline(ss, field, ',');
    p.vel.y = std::stod(field);
    std::getline(ss, field, ',');
    p.vel.z = std::stod(field);
    particles.push_back(p);
  }
  return particles;
}

void write_vtk(const std::string& path, const std::vector<Particle>& particles) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot open output file: " + path);

  out << "# vtk DataFile Version 3.0\n";
  out << "pepc-ttg particle snapshot\n";
  out << "ASCII\n";
  out << "DATASET POLYDATA\n";
  out << "POINTS " << particles.size() << " double\n";
  for (const auto& p : particles) out << p.pos.x << ' ' << p.pos.y << ' ' << p.pos.z << '\n';
  out << "POINT_DATA " << particles.size() << "\n";
  out << "SCALARS mass double 1\nLOOKUP_TABLE default\n";
  for (const auto& p : particles) out << p.mass << '\n';
  out << "VECTORS velocity double\n";
  for (const auto& p : particles) out << p.vel.x << ' ' << p.vel.y << ' ' << p.vel.z << '\n';
}

}  // namespace bh
