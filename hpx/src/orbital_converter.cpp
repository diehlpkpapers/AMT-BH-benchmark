#include "kepler_to_cartesian.h"
#include <filesystem>
#include <iostream>


// To build (from repo root): g++ -std=c++17 -o orbital_converter src/orbital_converter.cpp src/kepler_to_cartesian.cpp -I./include
// Converts two Kepler-elements CSVs to state-vector CSVs and combines them

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc != 4) {  // expect 3 args
        std::cerr << "Usage: " << argv[0]
                  << " <elements1.csv> <elements2.csv> <output_dir>\n";
        return 1;
    }

    fs::path in1    = argv[1];
    fs::path in2    = argv[2];
    fs::path outDir = argv[3];

    if (!fs::exists(outDir) || !fs::is_directory(outDir)) {
        std::cerr << "Error: output_dir must be a valid directory\n";
        return 1;
    }

    // derive intermediate names
    auto sv1 = outDir / (in1.stem().string() + "_sv.csv");
    auto sv2 = outDir / (in2.stem().string() + "_sv.csv");

    // convert inputs to state vector csvs
    if (!convertOrbitalElementsToCSV(in1.string(), sv1.string()) ||
        !convertOrbitalElementsToCSV(in2.string(), sv2.string())) {
        std::cerr << "Error: conversion failed\n";
        return 1;
    }

    // combines the csvs
    auto combined = outDir / (in1.stem().string()
                              + "_" + in2.stem().string()
                              + "_combined.csv");
    combineCSVFiles(sv1.string(), sv2.string(), combined.string());

    std::cout << "Done: " << combined << "\n";
    return 0;
}

