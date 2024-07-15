# Deep Percolation Events Analysis

This folder contains Python scripts and Jupyter Notebooks for analyzing deep percolation events in lysimeters, along with associated meteorological and soil data. The analysis includes event identification, characterization, visualization, and a literature review component to assess trends in percolation, runoff, and evapotranspiration research.

## Project Structure

* **Analysis:**
    * `event_identification.ipynb` (Jupyter Notebook): This notebook reads raw data (percolation, meteorological, soil) and processes it to identify and characterize deep percolation events in different lysimeters.
    * `analysis.ipynb` (Jupyter Notebook): This notebook performs statistical analysis on the processed data, comparing events across seasons, and investigating correlations between percolation and other factors. It generates visualizations to aid in understanding the results.
    * `user_functions.py`: This script contains custom functions used in the analysis, including data loading, event identification algorithms, statistical calculations, and plotting functions.
* **Literature Review:**
    * `lit_search_plot.ipynb`: This notebook reads results from literature searches on percolation, runoff, and evapotranspiration. It generates a line plot visualizing the trends in the number of published articles over time and a bar plot of total article counts.

## Data Requirements

* **Deep Percolation Data:** Hourly deep percolation measurements from lysimeters.
* **Meteorological Data:** Hourly and daily meteorological data (rainfall, snow depth, air temperature, etc.).
* **Soil Data:** Half-hourly soil moisture, soil temperature, and soil suction data.
* **Literature Search Results:** CSV files containing results from literature searches (e.g., Scopus results with year and article count).

## Dependencies

* Python 3.x
* pandas
* NumPy
* SciPy
* pingouin
* Matplotlib
* Seaborn
* Plotly
* Jupyter Notebook (for running the `.ipynb` files)

## Note
It is not necessary to use Jupyter Notebooks for running the analysis. One can simply copy the codes within the code cells in the notebooks and run them in a Python script or interactive Python environment.

## Usage

1. **Data Preparation:**
   * Download lysimeter, meteorological, and soil data from the Dataverse repository and place it in the `../data/` directory.
   * Place your literature search results (CSV files) in the project directory.
2. **Event Identification and Analysis:**
   * Open and run `event_identification.ipynb` to process the raw data and identify percolation events.
   * Open and run `analysis.ipynb` to perform statistical analysis and generate visualizations.
3. **Literature Review:**
   * Run `lit_search_plot.py` to generate the plot of literature search trends.

## Output

* The `event_identification.ipynb` notebook will save processed event data to the data directory.
* The `analysis.ipynb` notebook will generate plots and save them in a specified directory (e.g., `../Figures/`).
* The `lit_search_plot.ipynb` script will generate a PDF plot of literature search trends.
