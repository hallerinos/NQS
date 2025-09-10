// #include <torch/extension.h>
#include <pybind11/pybind11.h>

// Declare the roll_call_launcher function
void mlauncher();

// Write the C++ function that we will call from Python
void roll_call_binding() {
    mlauncher();
}

PYBIND11_MODULE(cusolve, m) {
  m.def(
    "roll_call", // Name of the Python function to create
    &roll_call_binding, // Corresponding C++ function to call
    "Launches the kernel" // Docstring
  );
}