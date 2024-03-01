function generate_outputs(model,spec,depths,iterations)
    #depth = zeros(iterations)
    #trophic_level = zeros(model.n_species+model.n_pool)
    #daily_ration = copy(trophic_level)
    mortalities = zeros(model.n_species,2)
    biomass = zeros(iterations,depths,model.n_species)
    consumption = zeros(Float64, spec, spec, depths,iterations)
    #consumption_biomass = zeros(model.n_species,iterations)
    production = zeros(iterations,model.n_species)
    #production_biomass = zeros(model.n_species,iterations)
    pool_density = zeros(iterations,model.n_pool)
    diet_composition = fill(NaN, (model.n_species+model.n_pool, model.n_species+model.n_pool))

    #timestep_array = Array{Float64,3}(undef,sum(model.ninds),5,iterations)
    #daily_array = Array{Float64,4}(undef,sum(model.ninds),4,model.n_iteration/1440). Fix later


    return MarineOutputs(mortalities,biomass,consumption,production,pool_density,diet_composition)

    #Holder function to combine outputs in one struct as in the individuals.
end

function calculate_trophic_levels(model, output, diet_matrix::Matrix{Float64}; max_iterations = 1e5, convergence_threshold = 1e-5)

    num_total = model.n_species + model.n_pool
    #Identify known trophic levels from pools. All of these will be at the end of the diet matrix
    known_trophic_levels = Dict(zip(collect((model.n_species+1):num_total),model.pools.pool.pool1.characters.Trophic_Level[2]))

    # Initialize trophic levels array
    trophic_levels = zeros(Float64, num_total)
        
    # Set trophic levels for known species
    for (species, level) in known_trophic_levels
        trophic_levels[species] = level
    end    

    # Iterative update of trophic levels. Loop stops if it converges
    for iteration in 1:max_iterations
        prev_trophic_levels = copy(trophic_levels)

        for i in 1:num_total
            if !haskey(known_trophic_levels, i)
                prey_trophic_levels = trophic_levels .* diet_matrix[i,:]
                trophic_levels[i] = sum(prey_trophic_levels)
            end
        end

        # Check for convergence
        if maximum(abs.(trophic_levels - prev_trophic_levels)) < convergence_threshold
            break
        end
    end

    output.trophiclevel = trophic_levels
    return nothing
end

function calculate_daily_ration(model,output)
    for sp in eachindex(fieldnames(typeof(model.individuals.animals)))
        name = Symbol("sp"*string(sp))
        spec_array = getfield(model.individuals.animals, name)

        spec_ration = zeros(length(spec_array.data.daily_ration))
        spec_ration .= spec_array.data.daily_ration ./spec_array.data.weight
        output.daily_ration[sp] = mean(filter(!isnan,spec_ration)) * 100
    end

end

function results!(model,output)

    calculate_daily_ration(model,output)
    calculate_trophic_levels(model,output,diet_matrix)

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

function depth_density(model,i,depth_dens) #Calculate the number of individuals in each 1m depth bin
    depth_max = 2000 #Replace with maximum depth
    time = [i]; 
    density = DataFrame(t=repeat(time, depth_max),z=[1:1:depth_max;],n=zeros(depth_max))

    for sp in 1:model.n_species #Cycle through each species
        name = Symbol("sp"*string(sp))
        spec_array = getfield(model.individuals.animals, name)

        for j in 1:model.ninds[sp]
            z = Int(ceil(spec_array.data.z[j]))
            if z == 0
                z =1
            end
            density.n[z] += 1
        end
    end


    depth_dens.Time = vcat(depth_dens.Time,density.t)
    depth_dens.Depth = vcat(depth_dens.Depth,density.z)
    depth_dens.Number = vcat(depth_dens.Number,density.n)


    return nothing
end

function mortalities(model,i) #Running tab of dead individuals
    #Can complicate this with size & age classes

end

function foodweb_matrix(model)
    #Create two matricies
    ## Call at each predation event.
    ## 1) Biomass of IBM and Pool species in each depth bin
    ### - Matrix: Dim = Nspecies x Ndepth bin. In a 3D model, expand the Dimensionality
    ## 2) Consumption of biomass between predator and prey in each depth bin
    ### - Matrix: Dim = Nspecies x Nspecies x Ndepth bin. 3D model would again increase Dimensionality


end