# Data Directory for Deep Percolation Analysis

This directory contains the raw and processed data used for the analysis of deep percolation events in lysimeters. It is organized into subdirectories for different data types and stages of the analysis process.

## Data Files

### Raw Data

* **`DailyMeteo_201707_202212.csv`:** Daily meteorological data from July 2017 to December 2022.
* **`FieldHourlyDeepPercolation_3Enclosures_4Lysimeters_201806_202210.csv`:** Hourly deep percolation measurements from four lysimeters across three enclosures from June 2018 to October 2022. This is the primary data used to identify and analyze deep percolation events.
* **`FieldSoilMoisture_4Enclosures_4Lysimeters_201806_202210.parquet`:** Half-hourly soil moisture data from the same lysimeters and time period as the percolation data.
* **`FieldSoilSuction_4Enclosures_4Lysimeters_201806_202210.parquet`:** Half-hourly soil suction data (also known as matric potential) from the same lysimeters and time period.
* **`FieldSoilTemperature_4Enclosures_4Lysimeters_201806_202210.parquet`:** Half-hourly soil temperature data from the same lysimeters and time period. Soil temperature can affect water movement and percolation processes.
* **`HourlyMeteo_201707_202212.parquet`:** Hourly meteorological data from July 2017 to December 2022. 
* **`InputWater_SVSoutput_ensAve.csv`:** This file contains simulated water input at the surface,  from the Soil-Vegetation-Snow (SVS) model.

### Processed Data

* **`/Lysimeter_Objects/`:** This subdirectory contains processed data objects in Pickle format (`*.pkl`) that represent individual lysimeters and the identified deep percolation events within them.
    * `L1_events.pkl`, `L2_events.pkl`, `L3_events.pkl`, `L4_events.pkl`: Each file corresponds to a specific lysimeter (L1, L2, L3, L4) and stores information about the events, such as their timing, volume, and associated meteorological and soil conditions.

### Literature Review Data

* **`/Lit_Review/`:** This subdirectory stores data related to the literature review component of the analysis.
    * `et_Scopus_results.csv`: Results of a literature search on "evapotranspiration" or "ET" from Scopus, with yearly article counts.
    * `percolation_Scopus_results.csv`: Results of a literature search on "percolation" from Scopus, with yearly article counts.
    * `runoff_Scopus_results.csv`: Results of a literature search on "runoff" from Scopus, with yearly article counts.


## Usage

These data files are used as inputs for the Python scripts and Jupyter Notebooks in (`../Code/`). 