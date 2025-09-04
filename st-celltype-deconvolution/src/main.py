from utils.data_loader import load_data
from models.vae import VAE
from models.gcn import GCN

def main():
    # Load the single-cell and spatial transcriptomics data
    sc_data = load_data('./scvi-tools-DestVI/data/sc_lymph_node_preprocessed.h5ad')
    st_data = load_data('./scvi-tools-DestVI/data/st_lymph_node_preprocessed.h5ad')

    # cell_types = len(sc_data.obs['cell_type'].unique().tolist())

    # Initialize the VAE model
    # vae_model = VAE(input_dim=sc_data.shape[1])
    
    # # Train the VAE model on single-cell data
    # vae_model.train(sc_data.X)

    # # Initialize the GCN model
    # gcn_model = GCN(input_dim=st_data.shape[1])
    
    # # Perform cell type deconvolution using the trained VAE and spatial data
    # deconvolved_results = gcn_model.forward(st_data, vae_model)

    # # Output the results
    # print("Deconvolved Cell Type Results:")
    # print(deconvolved_results)

if __name__ == "__main__":
    main()