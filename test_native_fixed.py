import sys; sys.path.append('code'); from unified_benchmark import CuQuantumNative; s=CuQuantumNative(20); s.init_zero_state(); print('Init OK'); s.cleanup()
