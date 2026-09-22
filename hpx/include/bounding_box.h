#pragma once

#include "body.h"
#include "bounding_box.h"
#include "linear_octree.h"
#include "hpx_collectives.h"
#include <vector>
#include <limits>


// Axis-aligned bounding box
struct BoundingBox
{
    Position min{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity() };

    Position max{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity() };
};

HPX_IS_BITWISE_SERIALIZABLE(BoundingBox);


// Geometry helpers

double min_distance_sq(const BoundingBox& b1, const BoundingBox& b2);

BoundingBox compute_local_bbox(const std::vector<Position>& points);

BoundingBox compute_global_bbox(const BoundingBox& local_bb, hpxc::Ctx& ctx);


/**
 * @brief Calculates the squared minimum distance between a point and an axis-aligned bounding box.
 * @param x The x-coordinate of the point.
 * @param y The y-coordinate of the point.
 * @param z The z-coordinate of the point.
 * @param b The bounding box.
 * @return The squared distance. If the point is inside the box, the distance is 0.
 */
double pointBoxDistanceSq(double x, double y, double z, const BoundingBox& b);


/**
 * @brief Calculates the physical bounding box for a given octree cell.
 * @param cell_key The key (prefix and depth) of the octree cell.
 * @param global_bb The global bounding box of the entire simulation.
 * @return The calculated BoundingBox in physical coordinates.
 */
BoundingBox getBoundingBoxForCell(const OctreeKey& cell,
                                const BoundingBox& global_bb);
