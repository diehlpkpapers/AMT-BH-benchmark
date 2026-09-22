#include "kepler_to_cartesian.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <iomanip>
#include <unordered_set>

// Convert orbital elements to Cartesian state vectors
Body* convertKeplerToCartesian(const OrbitalElements& elem) {

    // calculate mu (in AU and day)
    // double mu = G * mass_sun; // AU^3/day^2
    double mu = elem.mu;
    // Convert angles from degrees to radians if necessary
    // std::cout << "cKtoC: " << "mu = " <<  mu << std::endl;
    double inclination = elem.inclination;
    double argOfPeriapsis = elem.argOfPeriapsis;
    double longOfAscNode = elem.longOfAscNode;
    double meanAnomaly = elem.meanAnomaly; // this assumes t0 = t1 do I need to fix this

    double desiredTime = 2451544.5; // Julian Date for J2000.0
    double deltaTime = desiredTime - elem.epoch; // Δt in days, elem.epoch is t0
    double meanMotion = sqrt(mu / pow(elem.semiMajorAxis, 3)); // Ensure a is in AU
    meanAnomaly += meanMotion * deltaTime;
    meanAnomaly = fmod(meanAnomaly, 2 * M_PI); // Normalize to [0, 2π]
    if (meanAnomaly < 0) meanAnomaly += 2 * M_PI;

    // Solve Kepler's Equation for Eccentric Anomaly (E)
    double E = meanAnomaly; // Initial guess
    const double tolerance = 1e-8;
    int maxIterations = 100;
    for (int i = 0; i < maxIterations; ++i) {
        double f = E - elem.eccentricity * sin(E) - meanAnomaly;
        double f_prime = 1 - elem.eccentricity * cos(E);
        double delta = -f / f_prime;
        E += delta;
        if (fabs(delta) < tolerance) {
            break;
        }
    }

    // Compute True Anomaly (ν)
    double trueAnomaly = 2 * atan2(
        sqrt(1 + elem.eccentricity) * sin(E / 2),
        sqrt(1 - elem.eccentricity) * cos(E / 2)
    );

    // Compute distance (r)
    double r = elem.semiMajorAxis * (1 - elem.eccentricity * cos(E));

    // Position in orbital plane
    double x_orb = r * cos(trueAnomaly);
    double y_orb = r * sin(trueAnomaly);

    // Velocity in orbital plane
    double h = sqrt(mu * elem.semiMajorAxis) / r;
    double vx_orb = h * (-sin(E));
    double vy_orb = h * sqrt(1 - elem.eccentricity * elem.eccentricity) * cos(E);

    // Rotate to inertial frame
    double cos_Omega = cos(longOfAscNode);
    double sin_Omega = sin(longOfAscNode);
    double cos_i = cos(inclination);
    double sin_i = sin(inclination);
    double cos_omega = cos(argOfPeriapsis);
    double sin_omega = sin(argOfPeriapsis);

    // Rotation matrix components
    double R11 = cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i;
    double R12 = -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i;
    // double R13 = sin_Omega * sin_i;
    double R21 = sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i;
    double R22 = -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i;
    // double R23 = -cos_Omega * sin_i;
    double R31 = sin_omega * sin_i;
    double R32 = cos_omega * sin_i;
    // double R33 = cos_i;

    // Position in inertial frame
    double x = R11 * x_orb + R12 * y_orb;
    double y = R21 * x_orb + R22 * y_orb;
    double z = R31 * x_orb + R32 * y_orb;

    // Velocity in inertial frame
    double vx = R11 * vx_orb + R12 * vy_orb;
    double vy = R21 * vx_orb + R22 * vy_orb;
    double vz = R31 * vx_orb + R32 * vy_orb;

    // Create and return the Body object
    Body* body = new Body;
    // body->mass = elem.mass;
    body->x = x;
    body->y = y;
    body->z = z;
    body->vx = vx;
    body->vy = vy;
    body->vz = vz;
    body->ax = 0.0;
    body->ay = 0.0;
    body->az = 0.0;

    return body;
}


bool convertOrbitalElementsToCSV(const std::string& inputFilename, const std::string& outputFilename) {
    std::ifstream inputFile(inputFilename);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening input file: " << inputFilename << std::endl;
        return false;
    }

    std::ofstream outputFile(outputFilename);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file: " << outputFilename << std::endl;
        return false;
    }

    // set precision in scientific notation
    outputFile << std::setprecision(std::numeric_limits<double>::max_digits10);

    std::string line;

    // Read the header line
    if (!std::getline(inputFile, line)) {
        std::cerr << "Empty file or error reading input file: " << inputFilename << std::endl;
        return false;
    }

    // Process the header to map column names to indices
    std::istringstream headerStream(line);
    std::vector<std::string> headers;
    std::string header;
    while (std::getline(headerStream, header, ',')) {
        // Remove surrounding quotes from each header
        if (!header.empty() && header.front() == '"' && header.back() == '"') {
            header = header.substr(1, header.size() - 2);
        }
        headers.push_back(header);
    }

    // Map headers to indices
    std::unordered_map<std::string, int> headerMap;
    for (size_t i = 0; i < headers.size(); ++i) {
        headerMap[headers[i]] = static_cast<int>(i);
    }
    // std::cerr << "Header Map:\n";
    // for (const auto& [key, value] : headerMap) {
    //     std::cerr << "  " << key << " -> " << value << "\n";
    // }

    outputFile << "id,name,class,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z\n";

    // keep track of Sun-centric state vectors for non-satellites
    std::unordered_map<std::string, Body> centralBodies;

    // Random number generator for albedo approximation
    std::random_device rd;
    std::mt19937 gen(rd());
    int id = 0;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;

        // Split the line into tokens
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }

        int tokens_size = static_cast<int>(tokens.size());
        // Initialize essential variables
        std::string name = "";  // Default to empty string for missing names
        std::string classType = "";
        double mass = 0.0;
        double albedo = 0.0;
        double diameter = 0.0;
        // bool hasEssentialFields = true;

        // Extract `name`
        if (headerMap.count("name") && headerMap["name"] < tokens_size ) {
            name = tokens[headerMap["name"]];
        }

        // Extract and validate `class`
        if (headerMap.count("class") && headerMap["class"] < tokens_size && !tokens[headerMap["class"]].empty()) {
            classType = tokens[headerMap["class"]];
        } else {
            std::cerr << "Missing class field for row. Skipping..." << std::endl;
            continue; // Skip if `class` is missing
        }

        // Determine central body (default to Sun if absent)
        std::string centralBody = "Sun";
        if (headerMap.count("central_body") && headerMap["central_body"] < tokens_size) {
            centralBody = tokens[headerMap["central_body"]];
        }

        // Extract or approximate `mass`
        if (headerMap.count("mass") && headerMap["mass"] < tokens_size  && !tokens[headerMap["mass"]].empty()) {
            mass = std::stod(tokens[headerMap["mass"]]);
        }

        // Extract or approximate `albedo`
        if (headerMap.count("albedo") && headerMap["albedo"] < tokens_size  && !tokens[headerMap["albedo"]].empty()) {
            albedo = std::stod(tokens[headerMap["albedo"]]);
        } else {
            std::cout << "Missing albedo for body: " << name << ". Approximating based on class...\n";
            std::uniform_real_distribution<> dis(0.1, 0.2); // Default range for unknown classes

            if (classType == "AMO" || classType == "APO" || classType == "ATE" || classType == "IEO" ||
                classType == "MCA" || classType == "PAA" || classType == "HYA" || classType == "AST") {
                dis = std::uniform_real_distribution<>(0.450, 0.550);
            } else if (classType == "IMB") {
                dis = std::uniform_real_distribution<>(0.030, 0.103);
            } else if (classType == "MBA") {
                dis = std::uniform_real_distribution<>(0.097, 0.203);
            } else if (classType == "OMB") {
                dis = std::uniform_real_distribution<>(0.197, 0.500);
            } else if (classType == "CEN") {
                dis = std::uniform_real_distribution<>(0.450, 0.750);
            } else if (classType == "TJN") {
                dis = std::uniform_real_distribution<>(0.188, 0.124); // Check if range is reversed
            } else if (classType == "TNO") {
                dis = std::uniform_real_distribution<>(0.022, 0.130);
            }
            /*
            else if (classType == "STA" || classType == "PLA" || classType == "DWA" || classType == "SAT") {
                // dont know the for these classes
                // dis = std::uniform_real_distribution<>(0.450, 0.550);
            }
            */
            else {
                std::cerr << "Unknown albedo range for class '" << classType << "'. Using default range.\n";
            }
            albedo = dis(gen);
        }

        // Extract or approximate `diameter`
        if (headerMap.count("diameter") && headerMap["diameter"] < tokens_size  && !tokens[headerMap["diameter"]].empty()) {
            diameter = std::stod(tokens[headerMap["diameter"]]); //should be in km for asteroids
        } else if (headerMap.count("H") && headerMap["H"] < tokens_size  && !tokens[headerMap["H"]].empty()) {
            double H = std::stod(tokens[headerMap["H"]]);
            diameter = 1329 * std::pow(albedo, -0.5) * std::pow(10, -0.2 * H); // in km
            std::cout << "Missing diameter for body: " << name << ". Approximated using albedo and H-magnitude.\n";
        }

        // Calculate mass if still missing
        if (mass == 0.0 && diameter > 0.0) {
            double rho;
            // Determine density (rho) based on albedo and asteroid type
            if (albedo < 0.1) {
                rho = 1380; // C-type (Chondrite) in kg/m³
            } else if (albedo <= 0.2) {
                rho = 2710; // S-type (Stony) in kg/m³
            } else {
                rho = 5320; // M-type (Nickel-Iron) in kg/m³
            }
            // Convert diameter to radius in meters
            double radius = (diameter / 2.0) * 1000.0; // Convert diameter (km) to radius (m)
            // Calculate mass using the formula: mass = (4/3) * π * r³ * ρ
            mass = (4.0 / 3.0) * M_PI * std::pow(radius, 3) * rho; // in kg
            // std::cout << "Approximated mass for body: " << name << " using diameter and density.\n";
        }

        // If mass is still zero
        if (mass == 0.0) {
            std::cerr << "Mass could not be approximated for body: " << name
                    << ". Check if diameter and other properties are valid.\n";
        }

        // Parse orbital elements
        OrbitalElements elem;
        if (headerMap.count("e") && headerMap["e"] < tokens_size  && !tokens[headerMap["e"]].empty() &&
            headerMap.count("a") && headerMap["a"] < tokens_size  && !tokens[headerMap["a"]].empty() &&
            headerMap.count("i") && headerMap["i"] < tokens_size  && !tokens[headerMap["i"]].empty() &&
            headerMap.count("om") && headerMap["om"] < tokens_size  && !tokens[headerMap["om"]].empty() &&
            headerMap.count("w") && headerMap["w"] < tokens_size  && !tokens[headerMap["w"]].empty() &&
            headerMap.count("ma") && headerMap["ma"] < tokens_size  && !tokens[headerMap["ma"]].empty() &&
            headerMap.count("epoch") && headerMap["epoch"] < tokens_size  && !tokens[headerMap["epoch"]].empty()) {
            elem.eccentricity = std::stod(tokens[headerMap["e"]]);
            elem.semiMajorAxis = std::stod(tokens[headerMap["a"]]);
            elem.inclination = std::stod(tokens[headerMap["i"]]) * M_PI / 180.0;
            elem.longOfAscNode = std::stod(tokens[headerMap["om"]]) * M_PI / 180.0;
            elem.argOfPeriapsis = std::stod(tokens[headerMap["w"]]) * M_PI / 180.0;
            elem.meanAnomaly = std::stod(tokens[headerMap["ma"]]) * M_PI / 180.0;
            elem.epoch = std::stod(tokens[headerMap["epoch"]]);

        } else {
            std::cerr << "Missing essential orbital elements for row: " << line << ". Skipping..." << std::endl;
            continue; // Skip if essential orbital elements are missing
        }

        // for calculating mu, if it's moon and we have its central body, use the it's central body's mass,
        // otherwise use Sun's mass
        const double G = 1.48812e-34; // Gravitational constant in AU^3 kg^-1 day^-2
        const double mass_sun = 1.98847e30;     // Mass of the Sun in kg
        double mu = (classType == "SAT" && centralBodies.count(centralBody))
                    ? G * centralBodies[centralBody].mass
                    : G * mass_sun;
        elem.mu = mu;

        // Convert orbital elements to Cartesian state vectors
        Body* body = convertKeplerToCartesian(elem);

        // If this is a moon, translate by its central body's state
        if (classType == "SAT" && centralBodies.count(centralBody)) {
            Body& parent = centralBodies[centralBody];
            body->x  += parent.x;
            body->y  += parent.y;
            body->z  += parent.z;
            body->vx += parent.vx;
            body->vy += parent.vy;
            body->vz += parent.vz;
        }
        // Write state vectors to the output file
        outputFile << id << ","
                   << name << ","    // name (can be empty string if missing)
                   << classType << "," // class
                   << mass << "," // mass
                   << body->x << ","    // pos_x
                   << body->y << ","    // pos_y
                   << body->z << ","    // pos_z
                   << body->vx << ","   // vel_x
                   << body->vy << ","   // vel_y
                   << body->vz << "\n"; // vel_z

        // If planet or dwarf planet, store the body for translating it's moons later,
        // we assume the planets and dwarf planets come first in the csv
        // so we should have a proper map before getting to the moons
        body->mass = mass;
        if (classType == "PLA" || classType == "DWA") {
            centralBodies[name] = *body;    // copy the Body into the map
        }

        id++;
        delete body;
    }

    inputFile.close();
    outputFile.close();

    return true;
}


void combineCSVFiles(const std::string& inputFile1, const std::string& inputFile2, const std::string& outputFile) {
    std::ifstream file1(inputFile1);
    std::ifstream file2(inputFile2);
    std::ofstream output(outputFile);

    if (!file1.is_open() || !file2.is_open() || !output.is_open()) {
        std::cerr << "Error: Unable to open one or more files!" << std::endl;
        return;
    }

    std::unordered_set<std::string> uniqueNames; // To track already included "name" values
    std::string line;
    int currentID = 0; // Continuous ID starting from 0

    // Read and write headers (assuming both files have the same headers)
    if (std::getline(file1, line)) {
        output << line << "\n"; // Write header to the output file
    } else {
        std::cerr << "Error: File 1 is empty or invalid!" << std::endl;
        return;
    }

    // insert Sun as the first entry
    uniqueNames.insert("Sun");
    output << currentID++ << ",Sun,STA,1.988469999999999977e+30,0,0,0,0,0,0\n";

    // Helper function to process a file
    auto processFile = [&](std::ifstream& file) {
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;

            // Split the line into tokens
            while (std::getline(iss, token, ',')) {
                tokens.push_back(token);
            }

            if (tokens.size() > 1) { // Ensure the line has at least two columns
                std::string name = tokens[1]; // "name" is the second column
                if (!name.empty() && uniqueNames.find(name) != uniqueNames.end()) {
                    std::cerr << "Duplicate found: " << name << "\n";
                } else {
                    if (!name.empty()) {
                        uniqueNames.insert(name);
                    }
                    tokens[0] = std::to_string(currentID++); // Update the ID column to be continuous
                    for (size_t i = 0; i < tokens.size(); ++i) {
                        output << tokens[i];
                        if (i < tokens.size() - 1) output << ","; // Add commas between tokens
                    }
                    output << "\n";
                }
            }
        }
    };

    // Process both files
    processFile(file1);
    // Skip header in File 2
    if (std::getline(file2, line)) {
        // Do nothing (skip the header)
    }
    processFile(file2);

    std::cout << "Files combined successfully into: " << outputFile << std::endl;

    file1.close();
    file2.close();
    output.close();
}