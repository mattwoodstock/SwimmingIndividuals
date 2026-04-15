# ===================================================================
# SwimmingIndividuals.jl
# ===================================================================
"""
    SwimmingIndividuals

A high-performance agent-based model (ABM) for simulating individual-based marine 
ecosystem dynamics. The model simulates the life histories, behaviors, and population 
dynamics of marine organisms with detailed bioenergetics, predation, and fishing impacts.

## Key Features
- **Individual-Based Modeling**: Emergent population dynamics from bottom-up agent decisions
- **Data-Driven Environment**: Complex, time-varying habitats from NetCDF files
- **Detailed Bioenergetics**: "Wisconsin"-style energy budget model
- **Flexible Behavior**: Supports multiple behavioral archetypes (DVM, diving, etc.)
- **Mechanistic Predation**: High-resolution agent-on-agent predation with background functional response
- **Fisheries Module**: Commercial and recreational fishing with realistic regulations
- **High-Performance Computing**: GPU acceleration via CUDA and KernelAbstractions

## Main Functions
- `runSI(sim::MarineSimulation)`: Main simulation driver
- `generate_individuals()`: Create agents
- `generate_environment!()`: Load environmental data
- `initialize_resources()`: Create resource patches

## Example Usage
```julia
using SwimmingIndividuals

# Load configuration files and setup
files = CSV.read("files.csv", DataFrame)
trait = Dict(pairs(eachcol(CSV.read(...))))

# Run simulation
include("model.jl")
