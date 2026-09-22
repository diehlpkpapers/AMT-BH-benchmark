#pragma once

#include "body.h"
#include "bounding_box.h"
#include "linear_octree.h"
#include <vector>
#include <cstdint>


// Computes a 63-bit Morton code from 21-bit integer coordinates for each axis.
uint64_t morton63(uint32_t xi, uint32_t yi, uint32_t zi);

// Calculates Morton codes for a vector of positions
std::vector<uint64_t> generateMortonCodes(const std::vector<Position>& raw, const BoundingBox& bb);

uint64_t mortonPrefix(uint64_t code, int depth);

uint64_t spread21(uint32_t v);
// Pulls the 21 x, y, or z bits back out of a 63-bit Morton word.
uint32_t compact21(uint64_t m);

// De-interleaves the top 3*depth bits of the prefix into (ix,iy,iz) integer coordinates.
void decodePrefix(uint64_t prefix, int depth, uint32_t &ix, uint32_t &iy, uint32_t &iz);

// Decodes a Morton key to the position of its minimum corner in normalized [0,1) space.
Position key_to_normalized_position(uint64_t prefix, int depth);
