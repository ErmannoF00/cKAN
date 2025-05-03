# cKAN

**KAN Inference GUI** is an interactive C++ application that allows users to visualize and run inference on Kolmogorov–Arnold Networks (KANs). This tool integrates a CUDA-based inference engine, Qt-based GUI, and ONNX model support, making it ideal for exploring interpretable neural network architectures in real time.

⚠️ **Note**: This is an ongoing project under active development. Many features are experimental or incomplete. Contributions and feedback are very welcome.

---

## 🚀 Features

- **Real-Time Inference**: Perform fast inference using a custom CUDA backend.
- **Visual Layer Outputs**: View the input, spline activations (`φ_i(x)`), and final output per layer.
- **ONNX Model Loading**: Import pre-trained KAN models exported from PyTorch/PyKAN.
- **Spline Plotting**: See the shape of each univariate function used in the model.
- **Interactive Inputs**: Use sliders or fields to dynamically change input values and observe outputs.
- **Network Evolution Viewer** *(Planned)*: A visualization tool to observe how the KAN evolves during training (e.g., spline function changes, activation maps).

---

## 🛠 Tech Stack

- **C++17**: Core application and architecture
- **Qt 6**: Graphical user interface
- **CUDA**: GPU acceleration for inference
- **ONNX Runtime / ONNX Parser**: Import pre-trained models
- **CMake**: Cross-platform build system

---

## 🗂 Project Structure (Planned)

```
cKAN/
├── src/
│   ├── main.cpp                # Entry point
│   ├── gui/                    # Qt UI components
│   ├── inference/              # CUDA-based KAN layers
│   ├── onnx_loader/            # ONNX model parsing tools
│   └── utils/                  # Utility functions
├── models/                    # Sample ONNX models
├── CMakeLists.txt             # Build configuration
├── README.md
└── LICENSE
```

---

## Setup Instructions (WIP)

### Prerequisites:
- NVIDIA GPU with CUDA Toolkit installed
- Qt 6 development libraries
- CMake >= 3.16
- C++17-compatible compiler

### Build:
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Run:
```bash
./KANViewer
```

---

## Example Model
Place your exported `.onnx` model in the `models/` folder. It should contain:
- `W1`, `W2` sparse matrices
- Parameters for spline-based `φ_i(x)` functions

---

## 🤝 Contributing
This project is in its early stages. If you're interested in helping shape a visual, interpretable deep learning framework, feel free to open an issue or submit a pull request.

---

## 📄 License
This project is released under the MIT License. See `LICENSE` for details.
