#pragma once

// Defines the available policies for load balancing
enum class LBPolicy {
    Octant,  // Simple geometric decomposition
    Hist256  // Histogram based partitioning
};

// Defines the available policies for force calculation data exchange
enum class FCPolicy {
    Tree,          // Exchange full trees will use this for validation
    // Bodies,        // Exchange bodies
    LET,           // Exchange Locally Essential Trees
    LET_DirectSum  // Exchange LETs, but calculate remote forces via direct summation.
};