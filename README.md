# EVSim: Electric Vehicle Mobility Simulation

This repository is dedicated to simulating the mobility behavior of electric vehicle (EV) users.
We provide tools and scripts to facilitate the simulation process using publicly available datasets.
The primary dataset used in this repository is the Geolife Trajectories GPS dataset, which is preprocessed
to extract relevant spatio-temporal information for EV-related research.


## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Downloading Geolife Data](#downloading-geolife-data)
- [Data Preprocessing](#data-preprocessing)
  - [Filtering Data](#filtering-data)
  - [Data Format](#data-format)
- [Simulation](#simulation)
- [Contributing](#contributing)
- [License](#license)

## Overview

The main purpose of this repository is to preprocess the Geolife Trajectories GPS dataset for Electric Vehicle (EV) mobility simulation. Our preprocessing pipeline includes downloading the Geolife dataset, filtering the data to extract crucial spatio-temporal information, and preparing it for EV simulation.

## Getting Started

### Prerequisites

Before you begin, make sure you have the following prerequisites installed:
- Python 3.x
- pandas
- NumPy
- NetworkX
- OSMnx
- rasterio
- zarr
- xarray
- rioxarray

### Downloading Geolife Data

To download the Geolife Trajectories GPS data, you can use the provided script.
The script allows you to specify various parameters such as the root folder, time range, and study modes to filter the data.
Here are the parameters:
- `root_folder`: The root folder where Geolife Trajectories data is stored.
- `study_modes`: A list of trajectory modes to include (e.g., 'car', 'taxi').
- `study_start_time`: The start time of the study period.
- `study_end_time`: The end time of the study period.
- `study_min_lat`, `study_min_lon`, `study_max_lat`, `study_max_lon`: Bounding box coordinates to filter data.

Note: If you have your own spatio-temporal data in a compatible format, you can skip this step.

## Data Preprocessing

### Filtering Data

The data preprocessing scripts filter the downloaded data to include only trajectories that fall within the specified time range,
match the selected modes (e.g., car and taxi), and are located within the defined bounding box.

### Data Format

The preprocessed data should be in a specific format with columns including 'id', 'longitude', 'latitude', and 'timestamp'.
This format is essential for the subsequent simulation steps.

## Simulation

The repository provides tools and scripts for simulating EV battery consumption based on the preprocessed data.
You can customize the simulation parameters and analyze the results for EV-related research.

## Contributing

Contributions to this repository are welcome.
If you have improvements or additional features to suggest, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
