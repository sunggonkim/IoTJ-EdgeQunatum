#include <iostream>
#include <string>
#include <cstring>
#include "simulator.hpp"

int main(int argc, char** argv) {
    int qubits = 22;
    int depth = 5;
    std::string circuit = "Random";
    std::string storage_path = "cpp_state_vector.bin";
    SimMode mode = SimMode::Tiered_Async;
    bool verify = false;
    bool force_mode = false;

    // Simple Argument Parser
    for(int i=1; i<argc; i++) {
        std::string arg = argv[i];
        if(arg == "--qubits") {
            if(i+1 < argc) qubits = std::stoi(argv[++i]);
        } else if(arg == "--layers" || arg == "--depth") {
            if(i+1 < argc) depth = std::stoi(argv[++i]);
        } else if(arg == "--circuit") {
            if(i+1 < argc) circuit = argv[++i];
        } else if(arg == "--sim-mode") {
            if(i+1 < argc) {
                 std::string m = argv[++i];
                 if(m == "blocking") mode = SimMode::Tiered_Blocking;
                 else if(m == "native") mode = SimMode::Native;
                 else if(m == "uvm") mode = SimMode::UVM;
                 else mode = SimMode::Tiered_Async;
            }
        } else if(arg == "--storage") {
           if(i+1 < argc) storage_path = argv[++i];
          } else if(arg == "--verify") {
              verify = true;
          } else if(arg == "--force-mode") {
              force_mode = true;
        }
    }

    std::cout << "Starting C++ EdgeQuantum with " << qubits << " Qubits, " << depth << " Layers/Depth" << std::endl;
    std::cout << "Circuit: " << circuit << std::endl;
    // Mode printed by Sim constructor
    std::cout << "Storage: " << storage_path << std::endl;
    
    // ... path logic ... 
    if (storage_path.find("/") == std::string::npos) {
        storage_path = "/mnt/nvme/skim/edgeQuantum/code/" + storage_path;
    }

    EdgeQuantumSim sim(qubits, storage_path, mode, force_mode);

    if (verify) {
        bool ok = sim.validate_hadamard();
        std::cout << (ok ? "VERIFY_PASS" : "VERIFY_FAIL") << std::endl;
        return ok ? 0 : 1;
    }
    
    // Dispatch Circuit & Measure Time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (circuit == "QV") sim.run_qv(depth);
    else if (circuit == "VQC") sim.run_vqc(depth);
    else if (circuit == "QSVM") sim.run_qsvm(depth); // Feature Dim = depth
    else if (circuit == "GHZ") sim.run_ghz();
    else if (circuit == "Random") sim.run_random(depth);
    else if (circuit == "VQE") sim.run_vqe(depth); // Ansatz layers
    else {
        std::cerr << "Unknown circuit: " << circuit << ". Defaulting to Random." << std::endl;
        sim.run_random(depth);
    }
    
    cudaDeviceSynchronize(); // Ensure GPU is done
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    
    std::cout << "Total Time: " << diff.count() << " s" << std::endl;

    return 0;
}
