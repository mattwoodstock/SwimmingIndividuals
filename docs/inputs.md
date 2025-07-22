# Model Inputs

The `SwimmingIndividuals.jl` model is highly flexible and is configured through a series of user-provided input files. These files define everything from the biological traits of the species to the physical environment and the rules of the fisheries. All input files are expected to be located in the `/inputs` directory of the project.

---

## The File Manifest (`files.csv`)

This is the master file that tells the model where to find all other input data. It is a simple CSV with two columns that allows you to easily switch between different sets of input files without changing the main model code.

| Column     | Description                                               |
|------------|-----------------------------------------------------------|
| `File`     | A unique key for the data file (e.g., `focal_trait`).     |
| `Destination` | The relative path to the corresponding file.           |

**Example File Manifest (`files.csv`):**

| File           | Destination                    |
|----------------|--------------------------------|
| `focal_trait`  | `inputs/focal_trait.csv`       |
| `resource_trait` | `inputs/resource_trait.csv` |
| `params`       | `inputs/params.csv`            |
| `grid`         | `inputs/grid.csv`              |
| `fisheries`    | `inputs/fisheries.csv`         |
| `environment`  | `inputs/environment.nc`        |

---

## Main Parameters (`params.csv`)

This file controls the high-level configuration of the simulation run.

| Name        | Description                                               |
|-------------|-----------------------------------------------------------|
| `numspec`   | Number of focal species to simulate.                      |
| `numresource` | Number of resource species to simulate.                |
| `nts`       | Total number of timesteps for the simulation.            |
| `model_dt`  | Duration of a single timestep (minutes).                 |
| `output_dt` | Frequency (in timesteps) at which to save outputs.       |
| `architecture` | Computational backend to use (`CPU` or `GPU`).        |
| `spinup`    | Number of timesteps for model burn-in before output.     |
| `plt_diags` | Boolean (1 or 0) for generating diagnostic plots.         |

---

## Trait Files (`focal_trait.csv` & `resource_trait.csv`)

These files contain all biological and behavioral parameters for the simulated species. Each row represents a species.

| Column       | Description                                               |
|--------------|-----------------------------------------------------------|
| `SpeciesLong`| Full name of the species.                                 |
| `Biomass`    | Initial biomass density.                                  |
| `Max_Size`   | Maximum length (mm) the species can attain.              |
| `LWR_a`, `LWR_b` | Length-weight relationship parameters.               |
| `MR_type`    | Metabolic rate model (e.g., 1 for standard, 2 for cetacean). |
| `Swim_velo`  | Swim speed parameter.                                     |
| `Type`       | Behavioral archetype (e.g., `dvm_strong`, `pelagic_diver`). |
| ...          | (Many other parameters for bioenergetics, predation, and behavior.) |

---

## Environmental Preferences (`envi_pref.csv`)

Defines habitat suitability for each species based on environmental variables.

| Column     | Description                                               |
|------------|-----------------------------------------------------------|
| `species`  | Species name (must match trait files).                    |
| `variable` | Environmental variable (`temp-surf`, `bathymetry`, etc.). |
| `pref_min` | Absolute minimum tolerated value.                         |
| `opt_min`  | Lower bound of optimal range.                             |
| `opt_max`  | Upper bound of optimal range.                             |
| `pref_max` | Absolute maximum tolerated value.                         |

---

## Fishery Regulations (`fisheries.csv`)

Defines fishing rules and quotas for the simulation.

| Column        | Description                                               |
|---------------|-----------------------------------------------------------|
| `FisheryName` | Name of the fishery.                                      |
| `Species`     | Targeted species.                                         |
| `Role`        | Role of species (`target` or `bycatch`).                 |
| `Quota`       | Total allowable catch (tonnes).                           |
| `StartDay`, `EndDay` | Fishing season boundaries.                         |
| `L50`, `Slope`| Gear selectivity parameters.                              |
| ...           | (Additional columns for area, size limits, etc.)         |

---

## Environmental Data (`environment.nc`)

A NetCDF file containing spatial and temporal environmental variables for the model domain. It supports flexible dimensions and variable types.

| Variable     | Dimensions | Description                                   |
|--------------|------------|-----------------------------------------------|
| `temp`       | 4D         | Full water column temperature grid.           |
| `temp-surf`  | 3D         | Sea surface temperature.                      |
| `salinity`   | 3D or 4D   | Salinity values.                              |
| `chl`        | 3D         | Surface chlorophyll concentration.            |
| `bathymetry` | 2D         | Seafloor depth.                               |

---

These input files form the backbone of your `SwimmingIndividuals.jl` simulation and allow extensive flexibility in configuring species traits, environmental drivers, and human impacts.
