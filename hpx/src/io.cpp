#include "io.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <iomanip>
#include <cstring>
#include <unordered_map>
#include <filesystem>
#include "tinyxml2.h"


bool readCSV(
    const std::string& filename,
    std::vector<uint64_t>& ids, // New parameter
    std::vector<double>& masses,
    std::vector<Position>& positions,
    std::vector<Velocity>& velocities,
    int body_count) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    // Clear vectors to ensure they are empty before reading new data
    ids.clear();
    masses.clear();
    positions.clear();
    velocities.clear();

    std::string line;
    // Skip the header line of the CSV file
    std::getline(file, line);

    uint64_t current_id = 0;

    while (std::getline(file, line)) {
        // Stop if the desired number of bodies has been read
        if (body_count > 0 && masses.size() >= static_cast<size_t>(body_count)) {
            break;
        }

        std::stringstream ss(line);
        std::string field;

        try {
            // Skip unused columns: id, name, class
            std::getline(ss, field, ','); // Skip id
            std::getline(ss, field, ','); // Skip name
            std::getline(ss, field, ','); // Skip class

            // Read the required data: mass, position, velocity
            double mass, px, py, pz, vx, vy, vz;

            std::getline(ss, field, ','); mass = std::stod(field);
            std::getline(ss, field, ','); px = std::stod(field);
            std::getline(ss, field, ','); py = std::stod(field);
            std::getline(ss, field, ','); pz = std::stod(field);
            std::getline(ss, field, ','); vx = std::stod(field);
            std::getline(ss, field, ','); vy = std::stod(field);
            std::getline(ss, field, ','); vz = std::stod(field);

            // Store the parsed values
            masses.push_back(mass);
            positions.push_back({px, py, pz});
            velocities.push_back({vx, vy, vz});

            ids.push_back(current_id++);

        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse line, skipping: \"" << line << "\"" << std::endl;
        }
    }

    return true;
}


void writeSnapshot(int rank, int vs_counter,
                   const std::vector<uint64_t>& local_ids,
                   const std::vector<double>& local_masses,
                   const std::vector<Position>& local_positions,
                   const std::vector<Velocity>& local_velocities,
                   const std::vector<Acceleration>& local_accelerations,
                   const std::string& vs_dir)
{
    using namespace tinyxml2;
    namespace fs = std::filesystem;

    int numPoints = static_cast<int>(local_masses.size());
    if (numPoints == 0) return;

    fs::path rank_dir = fs::path(vs_dir) / std::to_string(rank);
    fs::create_directories(rank_dir);
    std::string filename = (rank_dir / ("sim." + std::to_string(vs_counter) + ".vtp")).string();

    XMLDocument doc;
    XMLElement* vtkFile = doc.NewElement("VTKFile");
    vtkFile->SetAttribute("type", "PolyData");
    vtkFile->SetAttribute("version", "0.1");
    doc.InsertFirstChild(vtkFile);

    XMLElement* polyData = doc.NewElement("PolyData");
    vtkFile->InsertEndChild(polyData);

    XMLElement* piece = doc.NewElement("Piece");
    piece->SetAttribute("NumberOfPoints", numPoints);
    piece->SetAttribute("NumberOfVerts", numPoints);
    polyData->InsertEndChild(piece);

    // Points
    {
        XMLElement* points = doc.NewElement("Points");
        piece->InsertEndChild(points);
        XMLElement* da = doc.NewElement("DataArray");
        da->SetAttribute("type", "Float64");
        da->SetAttribute("Name", "position");
        da->SetAttribute("NumberOfComponents", 3);
        da->SetAttribute("NumberOfTuples", numPoints);
        da->SetAttribute("format", "ascii");
        std::ostringstream ss;
        for (const auto& p : local_positions) {
            ss << p.x << " " << p.y << " " << p.z << "\n";
        }
        da->InsertEndChild(doc.NewText(ss.str().c_str()));
        points->InsertEndChild(da);
    }

    // PointData
    {
        XMLElement* pd = doc.NewElement("PointData");
        piece->InsertEndChild(pd);

        // --- Using your exact helper lambda structure ---

        auto writeUInt64Array = [&](const char* name, const std::vector<uint64_t>& arr) {
            XMLElement* da = doc.NewElement("DataArray");
            da->SetAttribute("type", "UInt64"); // Handles uint64_t
            da->SetAttribute("Name", name);
            da->SetAttribute("NumberOfComponents", 1);
            da->SetAttribute("NumberOfTuples", static_cast<int>(arr.size()));
            da->SetAttribute("format", "ascii");
            std::ostringstream ss;
            for (auto v : arr) ss << v << "\n";
            da->InsertEndChild(doc.NewText(ss.str().c_str()));
            pd->InsertEndChild(da);
        };

        // Helper lambda to write a single integer value for all points
        auto writeIntScalar = [&](const char* name, int value, int count) {
            XMLElement* da = doc.NewElement("DataArray");
            da->SetAttribute("type", "Int32");
            da->SetAttribute("Name", name);
            da->SetAttribute("NumberOfComponents", 1);
            da->SetAttribute("NumberOfTuples", count);
            da->SetAttribute("format", "ascii");
            std::ostringstream ss;
            for (int i = 0; i < count; ++i) {
                ss << value << "\n";
            }
            da->InsertEndChild(doc.NewText(ss.str().c_str()));
            pd->InsertEndChild(da);
        };

        auto writeFloatArray = [&](const char* name, const std::vector<double>& arr, int comps = 1) {
            XMLElement* da = doc.NewElement("DataArray");
            da->SetAttribute("type", "Float64");
            da->SetAttribute("Name", name);
            da->SetAttribute("NumberOfComponents", comps);
            // Use arr.size() for general arrays, and numPoints for special cases
            int numTuples = (comps > 1 || arr.empty()) ? numPoints : static_cast<int>(arr.size());
            da->SetAttribute("NumberOfTuples", numTuples);
            da->SetAttribute("format", "ascii");
            std::ostringstream ss;
            if (comps == 1) {
                for (auto v : arr) ss << v << "\n";
            } else { // This specific logic for velocity
                 for (const auto& v : local_velocities) {
                     ss << v.x << " " << v.y << " " << v.z << "\n";
                 }
            }
            da->InsertEndChild(doc.NewText(ss.str().c_str()));
            pd->InsertEndChild(da);
        };

        writeIntScalar("rank", rank, numPoints);
        writeUInt64Array("body_id", local_ids);
        writeFloatArray("velocity", {}, 3); // Pass empty vector as payload is handled inside
        writeFloatArray("mass", local_masses, 1);

        {
            std::vector<double> acc_magnitude(numPoints);
            for (int i = 0; i < numPoints; i++) {
                double ax = local_accelerations[i].x;
                double ay = local_accelerations[i].y;
                double az = local_accelerations[i].z;
                acc_magnitude[i] = std::sqrt(ax * ax + ay * ay + az * az);
            }
            writeFloatArray("acceleration", acc_magnitude, 1);
        }
    }

    // Verts
    {
        XMLElement* verts = doc.NewElement("Verts");
        piece->InsertEndChild(verts);
        // Offsets
        XMLElement* da_offsets = doc.NewElement("DataArray");
        da_offsets->SetAttribute("type", "Int64");
        da_offsets->SetAttribute("Name", "offsets");
        da_offsets->SetAttribute("format", "ascii");
        std::ostringstream ss_offsets;
        for (int i = 1; i <= numPoints; i++) ss_offsets << i << " ";
        da_offsets->InsertEndChild(doc.NewText(ss_offsets.str().c_str()));
        verts->InsertEndChild(da_offsets);
        // Connectivity
        XMLElement* da_conn = doc.NewElement("DataArray");
        da_conn->SetAttribute("type", "Int64");
        da_conn->SetAttribute("Name", "connectivity");
        da_conn->SetAttribute("format", "ascii");
        std::ostringstream ss_conn;
        for (int i = 0; i < numPoints; i++) ss_conn << i << " ";
        da_conn->InsertEndChild(doc.NewText(ss_conn.str().c_str()));
        verts->InsertEndChild(da_conn);
    }
    doc.SaveFile(filename.c_str());
}

void updatePVDFile(const cxxopts::ParseResult& args,
                   int size,
                   int vs_counter,
                   double current_time,
                   const std::string& vs_dir)
{
    using namespace tinyxml2;

    std::string input_stem = std::filesystem::path(args["file"].as<std::string>()).stem().string();
    std::string dt_str = args["dt"].as<std::string>();
    std::string tend_str = args["tend"].as<std::string>();
    double theta = args["theta"].as<double>();
    int nbodies = args["bodies"].as<int>();
    std::string fc = args["fc"].as<std::string>();
    std::ostringstream oss_fname;
    oss_fname << input_stem
            << "_dt" << dt_str
            << "_tend" << tend_str
            << "_theta" << theta
            << "_bodies" << nbodies
            << "_fc" << fc
            << ".pvd";
    std::string pvdFilename = oss_fname.str();
    std::string pvdPath = (std::filesystem::path(vs_dir) / pvdFilename).string();

    XMLDocument doc;
    XMLElement* vtkFile = nullptr;
    XMLElement* collection = nullptr;

    // If this is the first visualization step, create new file
    if (vs_counter == 0) {
        vtkFile = doc.NewElement("VTKFile");
        vtkFile->SetAttribute("type", "Collection");
        doc.InsertFirstChild(vtkFile);
        collection = doc.NewElement("Collection");
        vtkFile->InsertEndChild(collection);
    } else {

        XMLError e = doc.LoadFile(pvdPath.c_str());
        if (e != XML_SUCCESS) {
            std::cerr << "Error: Could not load PVD file " << pvdPath << " on step " << vs_counter << std::endl;
            return;
        }
        // Append to the existing collection
        vtkFile = doc.FirstChildElement("VTKFile");
        if (vtkFile) {
            collection = vtkFile->FirstChildElement("Collection");
        }
        if (!collection) {
            std::cerr << "Error: PVD file " << pvdPath << " is corrupted." << std::endl;
            return;
        }
    }

    for (int r = 0; r < size; r++) {
        XMLElement* dataSet = doc.NewElement("DataSet");
        dataSet->SetAttribute("timestep", current_time);
        dataSet->SetAttribute("part", r);
        std::ostringstream fileRef;
        fileRef << r << "/sim." << vs_counter << ".vtp";
        dataSet->SetAttribute("file", fileRef.str().c_str());
        collection->InsertEndChild(dataSet);
    }
    doc.SaveFile(pvdPath.c_str());
}


void writeReceivedLETs(int rank, int vis_step,
                       const std::vector<NodeRecord>& received_nodes,
                       const std::vector<int>& recv_counts,
                       const BoundingBox& global_bb,
                       const std::string& vs_dir)
{
    using namespace tinyxml2;
    namespace fs = std::filesystem;

    int numNodes = static_cast<int>(received_nodes.size());
    if (numNodes == 0) return; // This rank received nothing, so don't write a file

    // Each rank writes to its own directory
    fs::path rank_dir = fs::path(vs_dir) / std::to_string(rank);
    fs::create_directories(rank_dir);
    std::string filename = (rank_dir / ("received_lets." + std::to_string(vis_step) + ".vtu")).string();

    XMLDocument doc;
    XMLElement* vtkFile = doc.NewElement("VTKFile");
    vtkFile->SetAttribute("type", "UnstructuredGrid");
    vtkFile->SetAttribute("version", "0.1");
    doc.InsertFirstChild(vtkFile);

    XMLElement* uGrid = doc.NewElement("UnstructuredGrid");
    vtkFile->InsertEndChild(uGrid);

    XMLElement* piece = doc.NewElement("Piece");
    piece->SetAttribute("NumberOfPoints", numNodes * 8);
    piece->SetAttribute("NumberOfCells", numNodes);
    uGrid->InsertEndChild(piece);

    XMLElement* cellData = doc.NewElement("CellData");
    piece->InsertEndChild(cellData);

    // Build the source_rank and other attribute arrays
    std::vector<int> source_ranks;
    source_ranks.reserve(numNodes);
    std::vector<double> let_masses;
    let_masses.reserve(numNodes);
    std::vector<int> receiving_ranks(numNodes, rank);

    int node_idx = 0;
    for (int r = 0; r < static_cast<int>(recv_counts.size()); ++r) {
        for (int i = 0; i < recv_counts[r]; ++i) {
            source_ranks.push_back(r);
            let_masses.push_back(received_nodes[node_idx++].mass);
        }
    }

    // Helper to write an integer array to the XML
    auto writeIntArray = [&](const char* name, const std::vector<int>& arr) {
        XMLElement* da = doc.NewElement("DataArray");
        da->SetAttribute("type", "Int32");
        da->SetAttribute("Name", name);
        da->SetAttribute("format", "ascii");
        std::ostringstream ss;
        for (auto v : arr) ss << v << " ";
        da->InsertEndChild(doc.NewText(ss.str().c_str()));
        cellData->InsertEndChild(da);
    };
     // Helper to write a double array to the XML
    auto writeDoubleArray = [&](const char* name, const std::vector<double>& arr) {
        XMLElement* da = doc.NewElement("DataArray");
        da->SetAttribute("type", "Float64");
        da->SetAttribute("Name", name);
        da->SetAttribute("format", "ascii");
        std::ostringstream ss;
        for (auto v : arr) ss << v << " ";
        da->InsertEndChild(doc.NewText(ss.str().c_str()));
        cellData->InsertEndChild(da);
    };

    writeIntArray("source_rank", source_ranks);
    writeDoubleArray("mass", let_masses);
    writeIntArray("receiving_rank", receiving_ranks);
    // Points: The 8 corners of each bounding box
    XMLElement* points = doc.NewElement("Points");
    piece->InsertEndChild(points);
    XMLElement* da_points = doc.NewElement("DataArray");
    da_points->SetAttribute("type", "Float64");
    da_points->SetAttribute("NumberOfComponents", 3);
    da_points->SetAttribute("format", "ascii");
    std::ostringstream ss_points;
    for (const auto& node : received_nodes) {
        BoundingBox bb = getBoundingBoxForCell({node.prefix, node.depth}, global_bb);
        ss_points << bb.min.x << " " << bb.min.y << " " << bb.min.z << "\n";
        ss_points << bb.max.x << " " << bb.min.y << " " << bb.min.z << "\n";
        ss_points << bb.min.x << " " << bb.max.y << " " << bb.min.z << "\n";
        ss_points << bb.max.x << " " << bb.max.y << " " << bb.min.z << "\n";
        ss_points << bb.min.x << " " << bb.min.y << " " << bb.max.z << "\n";
        ss_points << bb.max.x << " " << bb.min.y << " " << bb.max.z << "\n";
        ss_points << bb.min.x << " " << bb.max.y << " " << bb.max.z << "\n";
        ss_points << bb.max.x << " " << bb.max.y << " " << bb.max.z << "\n";
    }
    da_points->InsertEndChild(doc.NewText(ss_points.str().c_str()));
    points->InsertEndChild(da_points);

    // Cells: Topology defining how points form boxes
    XMLElement* cells = doc.NewElement("Cells");
    piece->InsertEndChild(cells);

    // Connectivity
    XMLElement* da_conn = doc.NewElement("DataArray");
    da_conn->SetAttribute("type", "Int64");
    da_conn->SetAttribute("Name", "connectivity");
    da_conn->SetAttribute("format", "ascii");
    std::ostringstream ss_conn;
    for (int i = 0; i < numNodes; ++i) {
        long long offset = i * 8;
        ss_conn << offset+0 << " " << offset+1 << " " << offset+3 << " " << offset+2 << " ";
        ss_conn << offset+4 << " " << offset+5 << " " << offset+7 << " " << offset+6 << " ";
    }
    da_conn->InsertEndChild(doc.NewText(ss_conn.str().c_str()));
    cells->InsertEndChild(da_conn);

    // Offsets
    XMLElement* da_offsets = doc.NewElement("DataArray");
    da_offsets->SetAttribute("type", "Int64");
    da_offsets->SetAttribute("Name", "offsets");
    da_offsets->SetAttribute("format", "ascii");
    std::ostringstream ss_offsets;
    for (int i = 1; i <= numNodes; i++) ss_offsets << i * 8 << " ";
    da_offsets->InsertEndChild(doc.NewText(ss_offsets.str().c_str()));
    cells->InsertEndChild(da_offsets);

    // Types (VTK_HEXAHEDRON = 12)
    XMLElement* da_types = doc.NewElement("DataArray");
    da_types->SetAttribute("type", "UInt8");
    da_types->SetAttribute("Name", "types");
    da_types->SetAttribute("format", "ascii");
    std::ostringstream ss_types;
    for (int i = 0; i < numNodes; ++i) ss_types << "12 ";
    da_types->InsertEndChild(doc.NewText(ss_types.str().c_str()));
    cells->InsertEndChild(da_types);

    doc.SaveFile(filename.c_str());
}


void updateReceivedLETPVDFile(const cxxopts::ParseResult& args,
                              int size,
                              int vs_counter,
                              double current_time,
                              const std::string& sanity_dir)
{
    using namespace tinyxml2;


    std::string input_stem = std::filesystem::path(args["file"].as<std::string>()).stem().string();
    std::string dt_str = args["dt"].as<std::string>();
    std::string tend_str = args["tend"].as<std::string>();
    double theta = args["theta"].as<double>();
    int nbodies = args["bodies"].as<int>();
    std::string fc = args["fc"].as<std::string>();
    std::ostringstream oss_fname;
    oss_fname << input_stem
            << "_dt" << dt_str
            << "_tend" << tend_str
            << "_theta" << theta
            << "_bodies" << nbodies
            << "_fc" << fc
            << "_received_lets.pvd";
    std::string pvdFilename = oss_fname.str();
    std::string pvdPath = (std::filesystem::path(sanity_dir) / pvdFilename).string();

    XMLDocument doc;
    XMLElement* vtkFile = nullptr;
    XMLElement* collection = nullptr;


    if (vs_counter == 0) {
        // Create the root elements for a new file
        vtkFile = doc.NewElement("VTKFile");
        vtkFile->SetAttribute("type", "Collection");
        doc.InsertFirstChild(vtkFile);
        collection = doc.NewElement("Collection");
        vtkFile->InsertEndChild(collection);
    }
    // Otherwise, load and append
    else {
        XMLError e = doc.LoadFile(pvdPath.c_str());
        if (e != XML_SUCCESS) {
            std::cerr << "Error: Could not load LET PVD file " << pvdPath << " on step " << vs_counter << std::endl;
            return;
        }

        // Find the existing collection to append to
        vtkFile = doc.FirstChildElement("VTKFile");
        if (vtkFile) {
            collection = vtkFile->FirstChildElement("Collection");
        }

        // If the file is corrupted and has no collection, we can't proceed
        if (!collection) {
            std::cerr << "Error: LET PVD file " << pvdPath << " is corrupted." << std::endl;
            return;
        }
    }

    for (int r = 0; r < size; r++) {
        XMLElement* dataSet = doc.NewElement("DataSet");
        dataSet->SetAttribute("timestep", current_time);
        dataSet->SetAttribute("part", r);
        std::ostringstream fileRef;
        fileRef << r << "/received_lets." << vs_counter << ".vtu";
        dataSet->SetAttribute("file", fileRef.str().c_str());
        collection->InsertEndChild(dataSet);
    }
    doc.SaveFile(pvdPath.c_str());
}



/**
 * @brief Writes the global domain decomposition histogram to a single VTU file.
 *        This should only be called by rank 0.
 * @param vis_step The current visualization step number.
 * @param hist A vector of pairs, where each pair contains a bin's particle count and its assigned rank.
 * @param global_bb The global bounding box of the simulation, for coordinate mapping.
 * @param out_dir The directory to write the file to.
 */
void writeHistogram(int vis_step,
                    const std::vector<std::pair<long long, int>>& hist,
                    const BoundingBox& global_bb,
                    const std::string& out_dir,
                    int bucket_bits)
{
    using namespace tinyxml2;
    namespace fs = std::filesystem;

    int bucket_depth = bucket_bits / 3;
    // Collect data only for the non-empty histogram cells
    std::vector<BoundingBox> cell_boxes;
    std::vector<long long> cell_counts;
    std::vector<int> cell_ranks;

    for (size_t i = 0; i < hist.size(); ++i) {
        if (hist[i].first > 0) { // .first is particle_count
            uint64_t prefix = static_cast<uint64_t>(i) << (63 - bucket_bits);
            OctreeKey key = {prefix, (uint8_t)bucket_depth};
            cell_boxes.push_back(getBoundingBoxForCell(key, global_bb));
            cell_counts.push_back(hist[i].first);
            cell_ranks.push_back(hist[i].second); // .second is assigned_rank
        }
    }

    int numCells = static_cast<int>(cell_boxes.size());
    if (numCells == 0) return;

    fs::path hist_data_dir = fs::path(out_dir) / "histograms";
    fs::create_directories(hist_data_dir);
    std::string filename = (hist_data_dir / ("histogram." + std::to_string(vis_step) + ".vtu")).string();

    // Write the VTK Unstructured Grid file
    XMLDocument doc;
    XMLElement* vtkFile = doc.NewElement("VTKFile");
    vtkFile->SetAttribute("type", "UnstructuredGrid");
    vtkFile->SetAttribute("version", "0.1");
    doc.InsertFirstChild(vtkFile);

    XMLElement* uGrid = doc.NewElement("UnstructuredGrid");
    vtkFile->InsertEndChild(uGrid);

    XMLElement* piece = doc.NewElement("Piece");
    piece->SetAttribute("NumberOfPoints", numCells * 8);
    piece->SetAttribute("NumberOfCells", numCells);
    uGrid->InsertEndChild(piece);

    // CellData
    XMLElement* cellData = doc.NewElement("CellData");
    piece->InsertEndChild(cellData);

    auto writeIntArray = [&](const char* name, const std::vector<int>& arr) {
        XMLElement* da = doc.NewElement("DataArray");
        da->SetAttribute("type", "Int32");
        da->SetAttribute("Name", name);
        da->SetAttribute("format", "ascii");
        std::ostringstream ss;
        for (auto v : arr) ss << v << " ";
        da->InsertEndChild(doc.NewText(ss.str().c_str()));
        cellData->InsertEndChild(da);
    };
    auto writeLongArray = [&](const char* name, const std::vector<long long>& arr) {
        XMLElement* da = doc.NewElement("DataArray");
        da->SetAttribute("type", "Int64");
        da->SetAttribute("Name", name);
        da->SetAttribute("format", "ascii");
        std::ostringstream ss;
        for (auto v : arr) ss << v << " ";
        da->InsertEndChild(doc.NewText(ss.str().c_str()));
        cellData->InsertEndChild(da);
    };

    writeLongArray("particle_count", cell_counts);
    writeIntArray("assigned_rank", cell_ranks);

    // Points (8 corners per box)
    XMLElement* points = doc.NewElement("Points");
    piece->InsertEndChild(points);
    XMLElement* da_points = doc.NewElement("DataArray");
    da_points->SetAttribute("type", "Float64");
    da_points->SetAttribute("NumberOfComponents", 3);
    da_points->SetAttribute("format", "ascii");
    std::ostringstream ss_points;
    for (const auto& bb : cell_boxes) {
        ss_points << bb.min.x << " " << bb.min.y << " " << bb.min.z << "\n";
        ss_points << bb.max.x << " " << bb.min.y << " " << bb.min.z << "\n";
        ss_points << bb.min.x << " " << bb.max.y << " " << bb.min.z << "\n";
        ss_points << bb.max.x << " " << bb.max.y << " " << bb.min.z << "\n";
        ss_points << bb.min.x << " " << bb.min.y << " " << bb.max.z << "\n";
        ss_points << bb.max.x << " " << bb.min.y << " " << bb.max.z << "\n";
        ss_points << bb.min.x << " " << bb.max.y << " " << bb.max.z << "\n";
        ss_points << bb.max.x << " " << bb.max.y << " " << bb.max.z << "\n";
    }
    da_points->InsertEndChild(doc.NewText(ss_points.str().c_str()));
    points->InsertEndChild(da_points);

    // Cells (connectivity, offsets, types)
    XMLElement* cells = doc.NewElement("Cells");
    piece->InsertEndChild(cells);

    XMLElement* da_conn = doc.NewElement("DataArray");
    da_conn->SetAttribute("type", "Int64");
    da_conn->SetAttribute("Name", "connectivity");
    da_conn->SetAttribute("format", "ascii");
    std::ostringstream ss_conn;
    for (int i = 0; i < numCells; ++i) {
        long long offset = i * 8;
        ss_conn << offset+0 << " " << offset+1 << " " << offset+3 << " " << offset+2 << " ";
        ss_conn << offset+4 << " " << offset+5 << " " << offset+7 << " " << offset+6 << " ";
    }
    da_conn->InsertEndChild(doc.NewText(ss_conn.str().c_str()));
    cells->InsertEndChild(da_conn);

    XMLElement* da_offsets = doc.NewElement("DataArray");
    da_offsets->SetAttribute("type", "Int64");
    da_offsets->SetAttribute("Name", "offsets");
    da_offsets->SetAttribute("format", "ascii");
    std::ostringstream ss_offsets;
    for (int i = 1; i <= numCells; i++) ss_offsets << i * 8 << " ";
    da_offsets->InsertEndChild(doc.NewText(ss_offsets.str().c_str()));
    cells->InsertEndChild(da_offsets);

    XMLElement* da_types = doc.NewElement("DataArray");
    da_types->SetAttribute("type", "UInt8");
    da_types->SetAttribute("Name", "types");
    da_types->SetAttribute("format", "ascii");
    std::ostringstream ss_types;
    for (int i = 0; i < numCells; ++i) ss_types << "12 ";
    da_types->InsertEndChild(doc.NewText(ss_types.str().c_str()));
    cells->InsertEndChild(da_types);

    doc.SaveFile(filename.c_str());
}

/**
 * @brief Creates or updates the PVD timeline file for the histogram visualization.
 */
void updateHistogramPVDFile(const cxxopts::ParseResult& args,
                            int vs_counter,
                            double current_time,
                            const std::string& out_dir)
{
    using namespace tinyxml2;

    std::string input_stem = std::filesystem::path(args["file"].as<std::string>()).stem().string();
    std::string dt_str = args["dt"].as<std::string>();
    std::string tend_str = args["tend"].as<std::string>();
    double theta = args["theta"].as<double>();
    int nbodies = args["bodies"].as<int>();
    std::string fc = args["fc"].as<std::string>();
    std::ostringstream oss_fname;
    oss_fname << input_stem << "_dt" << dt_str << "_tend" << tend_str
              << "_theta" << theta << "_bodies" << nbodies << "_fc" << fc
              << "_histogram.pvd";
    std::string pvdFilename = oss_fname.str();
    std::string pvdPath = (std::filesystem::path(out_dir) / pvdFilename).string();

    XMLDocument doc;
    XMLElement* collection = nullptr;

    if (vs_counter == 0) {
        XMLElement* vtkFile = doc.NewElement("VTKFile");
        vtkFile->SetAttribute("type", "Collection");
        doc.InsertFirstChild(vtkFile);
        collection = doc.NewElement("Collection");
        vtkFile->InsertEndChild(collection);
    } else {
        if (doc.LoadFile(pvdPath.c_str()) != XML_SUCCESS) {
            std::cerr << "Error: Could not load histogram PVD file on a later step: " << pvdPath << std::endl;
            return;
        }
        collection = doc.FirstChildElement("VTKFile")->FirstChildElement("Collection");
    }

    if (!collection) {
        std::cerr << "Error: Could not create or find collection in PVD file." << std::endl;
        return;
    }

    // Add a DataSet entry for the current timestep's single histogram file
    XMLElement* dataSet = doc.NewElement("DataSet");
    dataSet->SetAttribute("timestep", current_time);
    dataSet->SetAttribute("group", "");
    dataSet->SetAttribute("part", "0");
    std::ostringstream fileRef;
    fileRef << "histograms/histogram." << vs_counter << ".vtu";
    dataSet->SetAttribute("file", fileRef.str().c_str());
    collection->InsertEndChild(dataSet);

    doc.SaveFile(pvdPath.c_str());
}


void saveReferenceCSV(const std::string& dir, const std::vector<Position>& positions, const std::vector<uint64_t>& ids) {
    std::ofstream os(std::filesystem::path(dir) / "final_ref.csv");
    os << "id,x,y,z\n";
    for (size_t i = 0; i < positions.size(); ++i) {
        os << ids[i] << "," << positions[i].x << "," << positions[i].y << "," << positions[i].z << "\n";
    }
}

// MODIFIED: The load function now reads IDs correctly.
std::vector<Position> loadReferenceCSV(const std::string& dir, int num_bodies) {
    std::ifstream is(std::filesystem::path(dir) / "final_ref.csv");
    if (!is.is_open()) {
        throw std::runtime_error("Could not open reference CSV: " + dir + "/final_ref.csv");
    }

    std::string line;
    std::getline(is, line); // Skip header

    std::vector<Position> pos(num_bodies);
    while (std::getline(is, line)) {
        std::stringstream ss(line);
        std::string field;

        std::getline(ss, field, ',');
        uint64_t id = std::stoull(field);

        std::getline(ss, field, ',');
        double x = std::stod(field);

        std::getline(ss, field, ',');
        double y = std::stod(field);

        std::getline(ss, field, ',');
        double z = std::stod(field);

        if (id < static_cast<uint64_t>(num_bodies)) {
            pos[id] = {x, y, z};
        }
    }
    return pos;
}

// MODIFIED: This function now calculates the local sum and performs a global reduction.
double computeDistanceSum(const std::vector<Position>& local_pos, const std::vector<uint64_t>& local_ids,
    const std::vector<Position>& ref_pos, hpxc::Ctx& ctx) {
    // 1. Calculate the local sum of distances based on body IDs.
    double local_sum = 0.0;
    for (size_t i = 0; i < local_pos.size(); ++i) {
        uint64_t id = local_ids[i];
        if (id < ref_pos.size()) {
            const Position& p1 = local_pos[i];
            const Position& p2 = ref_pos[id];
            double dx = p1.x - p2.x;
            double dy = p1.y - p2.y;
            double dz = p1.z - p2.z;
            local_sum += std::sqrt(dx*dx + dy*dy + dz*dz);
        }
    }

    // 2. Reduce to the global sum (every rank gets the result; only rank 0
    //    actually reports it).
    return hpx::collectives::all_reduce(ctx.comm_error_sum, local_sum,
        std::plus<double>{}, hpxc::this_site_arg(ctx.rank), hpxc::generation_arg(ctx.gen_error_sum++)).get();
}

std::ofstream openEnergyLog(const std::string& out_dir) {
    std::ofstream os(std::filesystem::path(out_dir) / "energy.csv");
    os << "step,time_days,kinetic,potential,total,rel_drift\n";
    return os;
}

void appendEnergyRow(std::ofstream& os, int step, double t,
    double kinetic, double potential, double total, double rel_drift) {
    os << std::scientific << std::setprecision(17)
       << step << ',' << t << ',' << kinetic << ',' << potential << ','
       << total << ',' << rel_drift << '\n';
    os.flush();
}