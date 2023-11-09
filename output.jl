
function species_density(inds,g_frame)

    dens = DataFrame(Species = String[], Cell = Int[], Density = Float64[])
    for i in 1:length(unique(inds.Species))
        for j in 1:length(unique(inds.cell))

            subs = inds[(inds.Species .== unique(inds.Species)[i]) .& (inds.cell .== unique(inds.cell)[j]), :]

            sub_frame = g_frame[g_frame.cell .== unique(inds.cell)[j], :]

            density = nrow(subs)/sub_frame.Volume[1]

            new_row = Dict("Species" => unique(inds.Species)[i], "Cell" => unique(inds.cell)[j],"Density" => density)

            push!(dens,new_row)
        end
    end
    return dens
end

function biomass_density(inds,g_frame)

    dens = DataFrame(Species = String[], Cell = Int[], Density = Float64[])
    for i in 1:length(unique(inds.Species))
        for j in 1:length(unique(inds.cell))

            subs = inds[(inds.Species .== unique(inds.Species)[i]) .& (inds.cell .== unique(inds.cell)[j]), :]

            sub_frame = g_frame[g_frame.cell .== unique(inds.cell)[j], :]

            density = sum(subs.weight)/sub_frame.Volume[1]

            new_row = Dict("Species" => unique(inds.Species)[i], "Cell" => unique(inds.cell)[j],"Density" => density)

            push!(dens,new_row)
        end
    end
    return dens
end