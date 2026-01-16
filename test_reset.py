from numba import cuda; try: cuda.current_context().reset(); print('Reset OK'); except: print('Reset Failed')
