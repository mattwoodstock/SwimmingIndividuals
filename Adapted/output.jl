mutable struct PlanktonOutputWriter
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

function PlanktonOutputWriter(;dir = "./results",
    diags_prefix = "diags",
    plankton_prefix = "plankton",
    write_log = false,
    save_diags = false,
    save_plankton = false,
    plankton_include = (:x, :y, :z, :Sz),
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

return PlanktonOutputWriter(dir, write_log, save_diags, save_plankton, diags_file, plankton_file,
     plankton_include, plankton_iteration_interval, max_filesize, 1, 1)
end