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


function write_output!(writer::Union{MarineOutputWriter, Nothing}, model::MarineModel, ΔT)
    if isa(writer, Nothing)
        return nothing
    else
        if writer.write_log
            write_species_dynamics(model.t, model.individuals.phytos,
                                   writer.filepath, model.mode)
        end

        if writer.save_diags
            if model.iteration % diags.iteration_interval == 0.0
                if filesize(writer.diags_file) ≥ writer.max_filesize
                    start_next_diags_file(writer)
                end
                write_diags_to_jld2(diags, writer.diags_file, model.t, model.iteration,
                                    diags.iteration_interval, model.grid)
            end
        end

        if writer.save_plankton
            if model.iteration % writer.plankton_iteration_interval == 0.0
                if filesize(writer.plankton_file) ≥ writer.max_filesize
                    start_next_plankton_file(writer)
                end
                write_individuals_to_jld2(model.individuals.animals, writer.plankton_file, model.t,
                                          model.iteration, writer.plankton_include)
            end
        end
    end
end

function write_individuals_to_jld2(animals::NamedTuple, filepath, t, iter, atts)
    jldopen(filepath, "a+") do file
        file["timeseries/t/$iter"] = t
        for sp in keys(animals)
            spi = NamedTuple{atts}([getproperty(animals[sp].data, att) for att in atts])
            for att in atts
                file["timeseries/$sp/$att/$iter"] = Array(spi[att])
            end
        end
    end
end