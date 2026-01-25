import cupy as cp
import cuquantum
from cuquantum import custatevec as cusv
import numpy as np

def test_correctness():
    n_qubits = 2
    # Initialize |0...0>
    # 2^2 = 4 elements
    sv = cp.zeros(1 << n_qubits, dtype=cp.complex64)
    sv[0] = 1.0 + 0j
    
    handle = cusv.create()
    
    # Apply X on qubit 0 via Matrix
    # X matrix
    mat = cp.array([[0, 1], [1, 0]], dtype=cp.complex64)
    # Row major, device pointer
    
    print("Applying X Gate on Qubit 0 (Python)...")
    
    ws_size = cusv.apply_matrix_get_workspace_size(
        handle,
        cuquantum.cudaDataType.CUDA_C_32F,
        n_qubits,
        mat.data.ptr,
        cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW,
        0, # adjoint
        1, # nTargets
        0, # nControls
        cuquantum.ComputeType.COMPUTE_32F
    )
    print(f"Workspace Size: {ws_size}")
    
    if ws_size > 0:
        ws = cp.cuda.alloc(ws_size)
        ws_ptr = ws.ptr
    else:
        ws_ptr = 0
        
    targets = np.array([0], dtype=np.int32)
    controls = np.array([], dtype=np.int32)
    
    cusv.apply_matrix(
        handle,
        sv.data.ptr,
        cuquantum.cudaDataType.CUDA_C_32F,
        n_qubits,
        mat.data.ptr,
        cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW,
        0,
        targets.ctypes.data, # targets ptr? Or just list?
        1,
        0, # controls ptr
        0, # control vals
        0, # nControls
        cuquantum.ComputeType.COMPUTE_32F,
        ws_ptr,
        ws_size
    )
    
    cp.cuda.Stream.null.synchronize()
    
    res = sv.get()
    print("State Vector:")
    print(res)
    
    # Expect |0> -> |1>. Index 0=0, Index 1=1.
    if np.abs(res[1] - 1.0) < 1e-5:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    test_correctness()
