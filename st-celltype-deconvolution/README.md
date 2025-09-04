# st-celltype-deconvolution
This project implements a cell type deconvolution task using Variational Autoencoders (VAE) and Graph Convolutional Networks (GCN) to analyze spatial transcriptomics (st) data.

## Project Structure
```
st-celltype-deconvolution
├── src
│   ├── main.py          # Entry point for the application
│   ├── models
│   │   ├── vae.py      # Variational Autoencoder model definition
│   │   └── gcn.py      # Graph Convolutional Network model definition
│   ├── utils
│   │   └── data_loader.py # Data loading and preprocessing utilities
│   └── types
│       └── index.py    # Type definitions and interfaces
├── data
│   ├── sc_lymph_node_preprocessed.h5ad # Preprocessed single-cell transcriptomics data
│   └── st_lymph_node_preprocessed.h5ad # Preprocessed spatial transcriptomics data
├── requirements.txt     # Required Python libraries and dependencies
├── README.md            # Project documentation and usage instructions
└── .gitignore           # Files and directories to ignore in version control
```

## Installation
To install the required dependencies, run:
```
pip install -r requirements.txt
```

## Usage
1. Prepare your data in the `data` directory.
2. Run the main application:
```
python src/main.py
```

## Model Overview
- **Variational Autoencoder (VAE)**: This model is designed to learn a latent representation of the input data, which can be used for generating new samples or for downstream tasks such as classification.
- **Graph Convolutional Network (GCN)**: This model leverages the graph structure of the data to perform node classification and can effectively capture the relationships between different cell types in the spatial transcriptomics data.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.