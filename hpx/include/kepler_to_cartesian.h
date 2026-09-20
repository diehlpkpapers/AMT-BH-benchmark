#ifndef KEPLER_TO_CARTESIAN_H
#define KEPLER_TO_CARTESIAN_H
#include <string>

// Structure representing orbital elements of a celestial body
struct OrbitalElements {
    // double mass;
    double eccentricity;
    double semiMajorAxis;
    double inclination;
    double argOfPeriapsis;
    double longOfAscNode;
    double meanAnomaly;
    double epoch;
    double mu;   // gravitational parameter for this orbit
};

// structure representing a  body (using state vectors)
struct Body {
    int index;             // unique index of body
    double mass;        // mass of the body
    double x, y, z;     // position coordinates
    double vx, vy, vz;  // velocity components
    double ax, ay, az;  // acceleration components
    std::string name;   // name of the body
};

// Read
// Convert orbital elements to Cartesian state vectors
Body* convertKeplerToCartesian(const OrbitalElements& elem);

// Function to convert orbital elements CSV to state vector CSV
bool convertOrbitalElementsToCSV(const std::string& inputFilename, const std::string& outputFilename);

// Function to combine two CSV files into one, removing duplicates based on the "name" column
void combineCSVFiles(const std::string& inputFile1, const std::string& inputFile2, const std::string& outputFile);

#endif
