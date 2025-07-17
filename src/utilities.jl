# ===================================================================
# Core Data Structures
# ===================================================================

# Struct for a single species' agents and parameters
mutable struct plankton
    data::AbstractArray
    p::NamedTuple
end

# Container for all focal species
struct individuals
    animals::NamedTuple
end

# The `resource` struct is no longer needed for the grid-based system.
# mutable struct resource ...

# Temporary container for prey information during search
mutable struct PreyInfo
    Predator::Int
    Sp::Int
    Ind::Int
    Type::Int
    Length::Float64
    Biomass::Float64
    Distance::Float64
end

# Defines the gear selectivity for a species within a fishery
struct Selectivity
    species::String
    L50::Float64
    slope::Float64
end

# Defines a single fishery's regulations and properties
mutable struct Fishery
    name::String
    target_species::Vector{String}
    bycatch_species::Vector{String}
    selectivities::Dict{String, Selectivity}
    quota::Float64
    cumulative_catch::Float64
    cumulative_inds::Int64
    season::Tuple{Int, Int}
    area::Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}, Tuple{Float64, Float64}}
    slot_limit::Tuple{Float64, Float64}
    bag_limit::Int64
end

# A flexible struct to hold any environmental data loaded from a NetCDF file
mutable struct MarineEnvironment
    data::Dict{String, AbstractArray}
    ts::Int # To hold the current month index
end

# Holds vertical distribution data for agents
mutable struct MarineDepths
    focal_day::DataFrame
    focal_night::DataFrame 
    patch_day::DataFrame
    patch_night::DataFrame
    grid::DataFrame
end

# The main model struct, containing the entire state of the simulation
mutable struct MarineModel
    arch::Architecture
    environment::MarineEnvironment
    depths::MarineDepths
    fishing::Vector{Fishery}
    t::Float64
    iteration::Int64
    dt::Float64
    individuals::individuals
    resources::NamedTuple{(:biomass, :capacity), Tuple{AbstractArray{Float64, 4}, AbstractArray{Float64, 4}}}
    resource_trait::DataFrame
    capacities::AbstractArray
    ninds::Int64
    n_species::Int64
    n_resource::Int64
    abund::Vector{Int64}
    bioms::Vector{Float64}
    init_abund::Vector{Int64}
    files::DataFrame
    output_dt::Int64
    spinup::Int64
end


# ===================================================================
# Helper and Utility Functions
# ===================================================================

# --- Spatial Indexing Helpers ---
@inline function get_cell_id(x, y, z, lonres, latres)
    return x + (y - 1) * lonres + (z - 1) * lonres * latres
end

@inline function get_cell_xyz(id, lonres, latres)
    z = div(id - 1, lonres * latres) + 1
    rem_id = (id - 1) % (lonres * latres)
    y = div(rem_id, lonres) + 1
    x = rem_id % lonres + 1
    return x, y, z
end

# --- Statistical and Distribution Helpers ---
function gaussmix(n, m1, m2, m3, s1, s2, s3, l1, l2)
    I = rand(n) .< l1
    I2 = rand(n) .< l1 .+ l2
    z = [rand(I[i] ? Normal(m1, s1) : (I2[i] ? Normal(m2, s2) : Normal(m3, s3))) for i in 1:n]
    return z
end

function multimodal_distribution(x, means, stds, weights)
    pdf_values = [weights[i] * pdf(Normal(means[i], stds[i]), x) for i in 1:length(means)]
    return sum(pdf_values)
end

function get_target_z(sp, dist)
    return gaussmix(1, dist[sp, "mu1"], dist[sp, "mu2"], dist[sp, "mu3"], dist[sp, "sigma1"], dist[sp, "sigma2"], dist[sp, "sigma3"], dist[sp, "lambda1"], dist[sp, "lambda2"])[1]
end

function lognormal_params_from_maxsize(max_size::Real)
    median_val = 1/3 * max_size
    percentile = 0.95
    μ = log(median_val)
    z_score = quantile(Normal(0, 1), percentile)
    σ = (log(max_size) - μ) / z_score
    return μ, σ
end

function lognormal_params_from_minmax(min_size::Real, max_size::Real)
    min_size = max(min_size, 1e-6)
    if min_size >= max_size; max_size = min_size * 1.1; end
    log_min, log_max = log(min_size), log(max_size)
    z = 1.96
    μ = (log_max + log_min) / 2.0
    σ = (log_max - log_min) / (2.0 * z)
    return μ, σ
end

# --- Geometric and Distance Helpers ---
function haversine(lat1, lon1, lat2, lon2)
    R = 6371000.0 # Earth radius in meters
    φ1, φ2 = deg2rad(lat1), deg2rad(lat2)
    Δφ = deg2rad(lat2 - lat1)
    Δλ = deg2rad(lon2 - lon1)
    a = sin(Δφ/2)^2 + cos(φ1) * cos(φ2) * sin(Δλ/2)^2
    c = 2 * atan(sqrt(a), sqrt(1 - a))
    return R * c
end

# --- Data Loading and Initialization Helpers ---
function generate_depths(files)
    focal_day = CSV.read(files[files.File .== "focal_z_dist_day", :Destination][1], DataFrame)
    focal_night = CSV.read(files[files.File .== "focal_z_dist_night", :Destination][1], DataFrame)
    resource_day = CSV.read(files[files.File .== "resource_z_dist_day", :Destination][1], DataFrame)
    resource_night = CSV.read(files[files.File .== "resource_z_dist_night", :Destination][1], DataFrame)
    grid = CSV.read(files[files.File .== "grid",:Destination][1], DataFrame)
    return MarineDepths(focal_day, focal_night, resource_day, resource_night, grid)
end

function load_ascii_raster(file_path::String)
    open(file_path, "r") do f
        header = Dict{String, Float64}()
        for _ in 1:6
            line = readline(f)
            key, val = split(line)
            header[key] = parse(Float64, val)
        end
        data = readdlm(f)
        return data
    end
end

# --- Physical Model Helpers ---
function ipar_curve(time, peak_ipar=450, peak_time=12, width=4)
    adj_time = time/60
    return peak_ipar * exp(-((adj_time - peak_time)^2) / (2 * width^2))
end