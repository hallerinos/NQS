#include <torch/extension.h>

// Declare the roll_call_launcher function
void mlauncher();

// Write the C++ function that we will call from Python
void roll_call_binding() {
    mlauncher();
}

void ai_launcher(torch::Tensor& in);

// Write the C++ function that we will call from Python
void array_increment_binding(torch::Tensor& in) {
    ai_launcher(in);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "array_increment", // Name of the Python function to create
    &array_increment_binding, // Corresponding C++ function to call
    "Launches the array_increment kernel" // Docstring
  );
}