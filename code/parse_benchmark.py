#!/usr/bin/env python3
"""Parse benchmark log and create JSON results"""
import re
import json

log_file = "data/benchmark_run_20260116_065453.log"

with open(log_file) as f:
    content = f.read()

results = []
current_qubits = None
current_circuit = None

# Parse patterns
qubit_pattern = r"Testing (\d+) Qubits"
circuit_pattern = r"Circuit: (\w+[-\d]*)"
result_pattern = r"(\w+.*?)\.\.\. [✅❌] ([\d.]+)s \((\d+) gates\)"
error_pattern = r"(\w+.*?)\.\.\. ❌ (.+)"

for line in content.split('\n'):
    qubit_match = re.search(qubit_pattern, line)
    if qubit_match:
        current_qubits = int(qubit_match.group(1))
        continue
    
    circuit_match = re.search(circuit_pattern, line)
    if circuit_match:
        current_circuit = circuit_match.group(1)
        continue
    
    result_match = re.search(result_pattern, line)
    if result_match and current_qubits and current_circuit:
        results.append({
            'qubits': current_qubits,
            'circuit': current_circuit,
            'simulator': result_match.group(1).strip(),
            'time': float(result_match.group(2)),
            'gates': int(result_match.group(3)),
            'success': True
        })

# Save JSON
with open("data/benchmark_results_parsed.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"Parsed {len(results)} results")

# Print summary table
print("\n" + "="*80)
print("BENCHMARK RESULTS SUMMARY")
print("="*80)

circuits = ['Hadamard', 'Random-10', 'QFT']
simulators = ['cuQuantum (Native)', 'cuQuantum (UVM)', 'BMQSim-like (Swap)', 'EdgeQuantum (Ours)']
qubits_list = [20, 22, 24, 26]

for circuit in circuits:
    print(f"\n📊 {circuit}:")
    header = f"{'Qubits':<8}"
    for sim in simulators:
        short_name = sim.split('(')[0].strip()[:12]
        header += f"{short_name:<14}"
    print(header)
    print("-" * 64)
    
    for q in qubits_list:
        row = f"{q:<8}"
        for sim in simulators:
            match = [r for r in results if r['qubits'] == q and r['circuit'] == circuit and r['simulator'] == sim]
            if match:
                row += f"{match[0]['time']:.2f}s".ljust(14)
            else:
                row += "N/A".ljust(14)
        print(row)

# Calculate speedups
print("\n" + "="*80)
print("SPEEDUP ANALYSIS (EdgeQuantum vs BMQSim-like)")
print("="*80)
print(f"{'Qubits':<8} {'Circuit':<12} {'BMQSim':<12} {'EdgeQuantum':<12} {'Speedup':<10}")
print("-" * 56)

for q in qubits_list:
    for circuit in circuits:
        bmq = [r for r in results if r['qubits'] == q and r['circuit'] == circuit and 'BMQSim' in r['simulator']]
        eq = [r for r in results if r['qubits'] == q and r['circuit'] == circuit and 'EdgeQuantum' in r['simulator']]
        if bmq and eq:
            speedup = bmq[0]['time'] / eq[0]['time']
            print(f"{q:<8} {circuit:<12} {bmq[0]['time']:.2f}s".ljust(32) + f"{eq[0]['time']:.2f}s".ljust(12) + f"{speedup:.1f}x")

print("\n✅ Results saved to: data/benchmark_results_parsed.json")
