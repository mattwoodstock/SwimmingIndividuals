mutable struct MarineOutputWriter
    filepath::String
    write_log::Bool
    save_diags::Bool
    save_plankton::Bool
    diags_file::String
    plankton_file::String
    plankton_include::Tuple
    plankton_iteration_interval::Int
    max_filesize::Number # in Bytes
    part_diags::Int
    part_plankton::Int
end

mutable struct MarineOutputs
    mortalities::Matrix{Float64}
    lengths::Vector{Float64}
    weights::Vector{Float64}
    consumption::Array{Float64,5}
    population_results::Array{Float64,3}
end

mutable struct MarineSimulation
    model::MarineModel                       # Model object
    ΔT::Float64                                  # model time step
    iterations::Int64                          # run the simulation for this number of iterations
    run::Int64                                  #Model run ID
    outputs::MarineOutputs
    #output_writer::Union{MarineOutputWriter,Nothing} # Output writer
end

function MarineOutputWriter(;dir = "./results",
    diags_prefix = "diags",
    plankton_prefix = "plankton",
    write_log = false,
    save_diags = false,
    save_plankton = false,
    plankton_include = (:x, :y, :z, :length),
    plankton_iteration_interval = 1,
    max_filesize = Inf
    )

isdir(dir) && rm(dir, recursive=true)
mkdir(dir)

diags_file = ""
plankton_file = ""

if save_diags
diags_file = joinpath(dir, diags_prefix*".jld2")
end
if save_plankton
plankton_file = joinpath(dir, plankton_prefix*".jld2")
end

return MarineOutputWriter(dir, write_log, save_diags, save_plankton, diags_file, plankton_file,
     plankton_include, plankton_iteration_interval, max_filesize, 1, 1)
end