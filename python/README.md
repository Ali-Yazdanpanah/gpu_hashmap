# Python wrapper (planned)

- **CuPy**: expose device pointers and call kernels from Python (CuPy raw kernels or custom CUDA).
- **PyBind11**: bind the C++ `gpu_hashmap` library and expose `insert_batch`, `lookup_batch` for use as a backend in distributed databases.

Build the C++ library first, then link from the Python extension.
