# ICONEEx Analysis

This repository contains Python scripts for processing and analyzing Eddy Covariance data from multiple tower locations in Ireland as part of the ICONeEx (Irish Carbon Observatory - Net Ecosystem Exchange) project.

## Project Overview

The ICONeEx Analysis project focuses on:
- Processing Eddy Covariance flux data using the `hesseflux` Python package
- Gap-filling and flux partitioning of Net Ecosystem Exchange (NEE) data
- Light Response Curve (LRC) analysis and parameter extraction
- Correlation analysis between flux parameters and satellite-derived Vegetation Indices
- Machine Learning models for NEE upscaling using satellite data
- Consolidated climatological dataset creation across multiple tower locations

## Tower Locations

The project includes data from the following tower sites:
- **Gurteen** - Grassland site
- **Athenry** - Agricultural site
- **JC1 & JC2** - Forest sites
- **Timoleague** - Coastal site
- **Lullymore** - Peatland site
- **Clarabog** - Bog site

## Repository Structure

```
iconeex_analysis/
├── src/
│   ├── Clarabog/                              # Clarabog tower processing scripts
│   ├── Lullymore/                             # Lullymore tower processing scripts
│   ├── Gurteen/                               # Gurteen tower and analysis scripts
│   ├── Biblio/                                # Bibliometric analysis scripts
│   ├── LCoPS/                                 # Land Cover and Policy Scenario scripts
│   └── consolidated_climatological_dataset/   # Multi-tower analysis scripts
├── GEMINI.md                                  # Detailed technical documentation
└── README.md                                 # This file
```

## Key Scripts

### Data Processing
- `src/*/process_*_data.py` - Tower-specific data processing scripts
- `src/consolidated_climatological_dataset/consolidate_data.py` - Multi-tower data consolidation

### Analysis Scripts
- `src/consolidated_climatological_dataset/generate_weekly_lrcs.py` - Weekly Light Response Curve generation
- `src/consolidated_climatological_dataset/analyze_lrc_parameter_correlations.py` - LRC parameter correlation analysis
- `src/consolidated_climatological_dataset/train_upscaling_model.py` - Machine Learning model training
- `src/consolidated_climatological_dataset/train_dl_upscaling_model.py` - Deep Learning model training

### Visualization
- Various plotting and visualization scripts across all directories
- Publication-quality figure generation scripts

## Dependencies

The main Python packages required for this project include:
- `hesseflux` - Eddy Covariance data processing (install from GitHub)
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `scikit-learn` - Machine Learning
- `xgboost` - Gradient boosting
- `tensorflow` - Deep Learning (with tensorflow-metal for Mac M1/M2)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/[your-username]/iconeex_analysis.git
cd iconeex_analysis
```

2. Install hesseflux from GitHub (required for NumPy compatibility):
```bash
git clone https://github.com/mcuntz/hesseflux.git
cd hesseflux
pip install .
cd ..
```

3. Install other dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn xgboost tensorflow tensorflow-metal
```

## Usage

Each script is designed to be run independently. The typical workflow is:

1. **Data Processing**: Run tower-specific processing scripts
2. **Consolidation**: Create consolidated climatological datasets
3. **Analysis**: Perform LRC analysis and correlation studies  
4. **Modeling**: Train machine learning models for upscaling

See `GEMINI.md` for detailed technical documentation and specific usage instructions.

## Model Performance

### Machine Learning Models (Latest Results)
- **XGBoost Model**: Training R² = 0.261, Testing R² = 0.351
- **Random Forest Model**: Training R² = 0.288, Testing R² = 0.358

### Deep Learning Models
Currently developing hybrid CNN-LSTM models for improved NEE upscaling performance.

## Data Structure

**Note**: This repository contains only the processing and analysis scripts. The actual data files are stored separately and are not version controlled due to their size and sensitivity.

Expected data structure (not included in repository):
```
[Data Directory]/
├── Gurteen/
│   ├── 2022.csv
│   ├── 2023.csv
│   └── processed_hesseflux_*.csv
├── [Other Tower Directories]/
└── consolidated climatological dataset/
    └── consolidated_half_hourly_climatology_data.csv
```

## Contributing

This is a research project repository. For questions or collaboration opportunities, please contact the project maintainers.

## License

This project is part of academic research. Please contact the authors for usage permissions and proper citation requirements.

## Citation

If you use this code in your research, please cite:
[To be updated with proper citation information]

## Contact

For questions about this codebase, please refer to the technical documentation in `GEMINI.md` or contact the project maintainers.
