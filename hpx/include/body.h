#pragma once

#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

// Basic vector for position, velocity, and acceleration
struct Vec3 {
    double x, y, z;
};

using Position = Vec3;
using Velocity = Vec3;
using Acceleration = Vec3;

HPX_IS_BITWISE_SERIALIZABLE(Vec3);