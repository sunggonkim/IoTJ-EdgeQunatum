"""
Run remaining C++ benchmarks for EdgeQuantum and BMQSim-like for qubits 30..34.
Appends results into `code/comprehensive_results.json`.
"""
import subprocess
import time
import json
import os
import datetime

CODE_DIR = os.path.dirname(__file__)
BINARY = os.path.join(CODE_DIR, 'build', 'edge_quantum')
RESULTS_FILE = os.path.join(CODE_DIR, 'comprehensive_results.json')

CIRCUITS = ["QV","VQC","QSVM","Random","GHZ","VQE"]
QUBITS = [30,31,32,33,34,35,36]
CIRCUITS = ["QV","VQC","QSVM","Random","GHZ","VQE"]
TARGET_SCHEMES = [
	{"name":"BMQSim-like (Offload)", "mode":"blocking", "force": True},
	{"name":"EdgeQuantum", "mode":"async", "force": True}
]

TIMEOUT_SEC = 3600

def load_results():
	if os.path.exists(RESULTS_FILE):
		with open(RESULTS_FILE,'r') as f:
			return json.load(f)
	return {"runs":[], "meta": {"start_time": datetime.datetime.now().isoformat()}}

def save_results(results):
	with open(RESULTS_FILE,'w') as f:
		json.dump(results, f, indent=2)

def already_run(results, scheme, circuit, qubits):
	# If environment forces rerun, always return False
	if os.environ.get('FORCE_RERUN','0')=='1':
		return False
	for r in results.get('runs',[]):
		if r.get('scheme')==scheme and r.get('circuit')==circuit and r.get('qubits')==qubits:
			return True
	return False

def run_cpp(qubits, circuit, mode, force):
	cmd = [BINARY, '--qubits', str(qubits), '--circuit', circuit, '--depth', '5']
	if mode and mode!='async':
		cmd += ['--sim-mode', mode]
	if force:
		cmd += ['--force-mode']
	start = time.time()
	try:
		res = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC)
		wall = time.time()-start
		if res.returncode != 0:
			return {"success": False, "error": f"Exit {res.returncode}", "wall_time": wall, "logs": res.stdout+res.stderr}
		sim_time = wall
		for line in res.stdout.splitlines():
			if 'Total Time:' in line:
				try:
					sim_time = float(line.split(':',1)[1].strip().split()[0])
				except:
					pass
		return {"success": True, "sim_time": sim_time, "wall_time": wall}
	except subprocess.TimeoutExpired:
		return {"success": False, "error": "TIMEOUT", "wall_time": TIMEOUT_SEC}
	except Exception as e:
		return {"success": False, "error": str(e), "wall_time": time.time()-start}

def main():
	print('Running remaining experiments for qubits', QUBITS)
	if not os.path.exists(BINARY):
		print('ERROR: binary not found at', BINARY)
		return
	results = load_results()
	for q in QUBITS:
		for c in CIRCUITS:
			for s in TARGET_SCHEMES:
				name = s['name']
				if already_run(results, name, c, q):
					print(f'SKIP {name} {c} {q} (already)')
					continue
				print(f'RUN {name} {c} {q}')
				r = run_cpp(q, c, s['mode'], s['force'])
				rec = {"scheme": name, "circuit": c, "qubits": q, "depth": 5, "timestamp": datetime.datetime.now().isoformat(), **r}
				results['runs'].append(rec)
				save_results(results)
				if r.get('success'):
					print(' SUCCESS time', r.get('sim_time'))
				else:
					print(' FAIL', r.get('error'))
				time.sleep(2)
	print('Done')

if __name__=='__main__':
	main()
