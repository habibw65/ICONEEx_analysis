"""
ERA5 Data Integration Workflow Setup

This script guides you through the complete workflow for integrating ERA5 reanalysis data
with your existing ICONeEx flux tower data for enhanced NEE upscaling.

Workflow Steps:
1. Download ERA5 data using CDS API
2. Process GRIB files and extract data for tower locations
3. Integrate with existing LRC parameters and satellite VIs
4. Train enhanced machine learning models
5. Evaluate and visualize results

Author: ICONeEx Analysis Team
"""

import os
import sys
import subprocess
from datetime import datetime

def check_cds_credentials():
    """Check if CDS API credentials are configured."""
    cdsrc_path = os.path.expanduser("~/.cdsapirc")
    
    if os.path.exists(cdsrc_path):
        print("✅ CDS API credentials found at ~/.cdsapirc")
        return True
    else:
        print("❌ CDS API credentials not found")
        print("Please set up your CDS API credentials:")
        print("1. Register at https://cds.climate.copernicus.eu/api-how-to")
        print("2. Create ~/.cdsapirc file with your credentials:")
        print("   url: https://cds.climate.copernicus.eu/api/v2")
        print("   key: YOUR_API_KEY")
        return False

def check_required_packages():
    """Check if required packages are installed."""
    required_packages = [
        'cdsapi', 'xarray', 'cfgrib', 'netcdf4', 'pandas', 
        'numpy', 'matplotlib', 'sklearn', 'xgboost'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\\n✅ All required packages are installed!")
        return True

def check_data_availability():
    """Check if required data files are available."""
    data_paths = {
        'Consolidated climatology data': "/Users/habibw/Documents/consolidated climatological dataset/consolidated_half_hourly_climatology_data.csv",
        'LRC parameters': "/Users/habibw/Documents/consolidated climatological dataset/all_towers_weekly_lrc_parameters.csv",
        'ERA5 download script': "/Users/habibw/Documents/iconeex_analysis/src/download_era5_sample.py",
        'ERA5 processing script': "/Users/habibw/Documents/iconeex_analysis/src/process_era5_data.py",
        'Enhanced ML script': "/Users/habibw/Documents/iconeex_analysis/src/era5_enhanced_nee_model.py"
    }
    
    all_available = True
    
    for name, path in data_paths.items():
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path}")
            all_available = False
    
    return all_available

def print_workflow_instructions():
    """Print detailed workflow instructions."""
    print("\\n" + "="*60)
    print("🚀 ERA5 DATA INTEGRATION WORKFLOW")
    print("="*60)
    
    print("\\n📋 COMPLETE WORKFLOW STEPS:")
    print("\\n1. 📥 DOWNLOAD ERA5 DATA:")
    print("   python3 /Users/habibw/Documents/iconeex_analysis/src/download_era5_sample.py")
    print("   ⏱️  This may take several hours depending on your data range")
    print("   💾 Data will be saved to: /Users/habibw/Documents/ERA5_Data/")
    
    print("\\n2. 🔧 PROCESS ERA5 GRIB FILES:")
    print("   python3 /Users/habibw/Documents/iconeex_analysis/src/process_era5_data.py")
    print("   ⚙️  Converts GRIB to CSV, extracts tower locations, calculates derived variables")
    print("   📊 Creates half-hourly timeseries matching flux tower data")
    
    print("\\n3. 🤖 TRAIN ENHANCED ML MODELS:")
    print("   python3 /Users/habibw/Documents/iconeex_analysis/src/era5_enhanced_nee_model.py")
    print("   🧠 Integrates ERA5 + LRC parameters + Satellite VIs")
    print("   📈 Trains XGBoost, Random Forest, and Deep Learning models")
    print("   📊 Generates comprehensive performance plots")
    
    print("\\n📦 DATA INTEGRATION INCLUDES:")
    print("   🌤️  ERA5 Meteorological Variables:")
    print("      • Air temperature & dewpoint temperature")
    print("      • Surface pressure & wind speed/direction")
    print("      • Solar radiation & precipitation")
    print("      • Vapor Pressure Deficit (VPD)")
    print("      • Boundary layer height & evaporation")
    
    print("   🛰️  Satellite Vegetation Indices:")
    print("      • EVI, NDVI, NDMI, SAVI")
    
    print("   💡 Light Response Curve Parameters:")
    print("      • Weekly LRC parameters (a, b, c)")
    print("      • R-squared values")
    
    print("   ⏰ Temporal Features:")
    print("      • Seasonal and diurnal cycles")
    print("      • Cyclical encoding for periodicity")

def estimate_processing_time():
    """Provide time estimates for each step."""
    print("\\n⏱️  ESTIMATED PROCESSING TIMES:")
    print("   1. ERA5 Download: 2-8 hours (depends on data range & CDS queue)")
    print("   2. ERA5 Processing: 10-30 minutes")
    print("   3. ML Model Training: 30-60 minutes")
    print("   📈 Total workflow: ~3-9 hours (mostly automated)")

def main():
    """Main setup function."""
    print("🌟 ERA5-Enhanced NEE Upscaling Setup")
    print("=" * 50)
    
    print("\\n🔍 Checking system requirements...")
    
    # Check CDS credentials
    cds_ok = check_cds_credentials()
    print()
    
    # Check packages
    packages_ok = check_required_packages()
    print()
    
    # Check data availability
    data_ok = check_data_availability()
    
    print("\\n" + "="*50)
    print("📊 SETUP SUMMARY")
    print("="*50)
    
    if cds_ok and packages_ok and data_ok:
        print("✅ All requirements satisfied! Ready to proceed.")
        print_workflow_instructions()
        estimate_processing_time()
        
        print("\\n🎯 QUICK START:")
        print("If you want to run a small test first, modify the date range in")
        print("download_era5_sample.py to just a few days for testing.")
        
    else:
        print("❌ Some requirements are missing. Please address the issues above.")
        
        if not cds_ok:
            print("\\n🔧 To set up CDS API credentials:")
            print("   1. Visit: https://cds.climate.copernicus.eu/api-how-to")
            print("   2. Register and get your API key")
            print("   3. Create ~/.cdsapirc with your credentials")
        
        if not packages_ok:
            print("\\n📦 To install missing packages:")
            print("   pip install -r requirements.txt")
        
        if not data_ok:
            print("\\n📊 Some data files are missing. Make sure you have:")
            print("   • Run the consolidated climatology data creation first")
            print("   • Generated LRC parameters")
            print("   • All scripts are in the correct locations")

if __name__ == "__main__":
    main()
