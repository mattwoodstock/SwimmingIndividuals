mutable struct MarineOutputs
    mortalities::Matrix{Float64}
    lengths::Vector{Float64}
    weights::Vector{Float64}
    #consumption::Array{Float64,5}
    #encounters::Array{Float64,5}
    #population_results::Array{Float64,3}
end

mutable struct MarineSimulation
    model::MarineModel                       # Model object
    ΔT::Float64                                  # model time step
    iterations::Int64                          # run the simulation for this number of iterations
    run::Int64                                  #Model run ID
    outputs::MarineOutputs
    #output_writer::Union{MarineOutputWriter,Nothing} # Output writer
end