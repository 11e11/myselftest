from typing import Any, Dict, List, Tuple

# Define the data structure for single-cell transcriptomics data
class SingleCellData:
    def __init__(self, expression_matrix: Any, cell_metadata: Dict[str, Any]):
        self.expression_matrix = expression_matrix
        self.cell_metadata = cell_metadata

# Define the data structure for spatial transcriptomics data
class SpatialTranscriptomicsData:
    def __init__(self, expression_matrix: Any, spatial_coordinates: List[Tuple[float, float]]):
        self.expression_matrix = expression_matrix
        self.spatial_coordinates = spatial_coordinates

# Define the interface for the cell type deconvolution model
class CellTypeDeconvolutionModel:
    def fit(self, single_cell_data: SingleCellData, spatial_data: SpatialTranscriptomicsData) -> None:
        pass

    def predict(self, spatial_data: SpatialTranscriptomicsData) -> Dict[str, Any]:
        pass