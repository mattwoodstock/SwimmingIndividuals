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
                write_individuals_to_jld2(model.individuals.animals, writer.plankton_file, model.t, model.iteration, writer.plankton_include)
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