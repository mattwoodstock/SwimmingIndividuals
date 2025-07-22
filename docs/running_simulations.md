## Running Simulations

Once the model components are initialized, the simulation is ready to be run. The core logic for managing and executing the time-stepping loop is handled by the `simulation.jl` and `update.jl` files. These files define the main simulation object and the function that drives the model forward in time.

### `simulation.jl`: Defining the Simulation State

This file defines the primary data structures that hold the simulation's state and its outputs.

* **`MarineOutputs`**: This struct is a container for all the large, multi-dimensional arrays that store the results of the simulation, such as mortality rates and consumption grids. Defining these in a separate struct keeps the main model object cleaner and easier to manage. The fields are defined as `AbstractArray` to ensure they are compatible with both standard CPU `Array`s and `CUDA.CuArray`s for GPU execution.

* **`MarineSimulation`**: This is the main "runner" object for a single simulation experiment. It bundles the complete `MarineModel` state with run-specific parameters like the timestep duration (`ΔT`), the total number of iterations, and the `MarineOutputs` container.

```julia
# In simulation.jl

# ===================================================================
# Data Structures for Simulation and Output
# ===================================================================

"""
    MarineOutputs
A mutable struct to hold all the data generated during a simulation run.
The fields are defined as `AbstractArray` to be compatible with both
CPU `Array`s and GPU `CuArray`s.
"""
mutable struct MarineOutputs
    mortalities::AbstractArray{Int64,5}
    Fmort::AbstractArray{Int64,5}
    consumption::AbstractArray{Float64,5}
    abundance::AbstractArray{Float64,4}
end


"""
    MarineSimulation
The main "runner" object for a single simulation experiment. It bundles
the model state with run-specific parameters
"""
mutable struct MarineSimulation
    model::MarineModel
    ΔT::Float64
    iterations::Int64
    run::Int64
    outputs::MarineOutputs
end
```

### update.jl: The Main Driver
This file contains the primary function that executes the simulation.

#### runSI(sim::MarineSimulation): This function is the main entry point for starting a simulation run. It takes a fully configured MarineSimulation object and executes the main time-stepping loop by calling the TimeStep! function for the specified number of iterations. It also prints messages to the console to indicate the start and completion of the run.

```julia
# In update.jl

# ===================================================================
# Main Simulation Driver
# ===================================================================

"""
    runSI(sim::MarineSimulation)

A simple wrapper function to start the simulation. This can be used as the
main entry point in your `model.jl` script.
"""
function runSI(sim::MarineSimulation)
    println("✅ Model Initialized. Starting simulation run...")
    for i in 1:sim.iterations
        TimeStep!(sim)
    end
    println("✅ Simulation run complete.")
end
```