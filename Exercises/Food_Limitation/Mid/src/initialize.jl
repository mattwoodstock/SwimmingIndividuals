using PlanktonIndividuals
using CSV, DataFrames, StructArrays, Distributions

#Read in grid and trait databases variables
state = CSV.read("state.csv",DataFrame)
trait = CSV.read("traits.csv",DataFrame)

# Create individuals
function generate_individuals(params, Nsp, grid_df,g::String)
    

    rawdata = DataFrame(ID = String[], Species = String[], x = Float64[], y = Float64[], z = Float64[],length = Float64[], weight = Float64[], energy = Float64[], age = Float64[], generation = Int[], move = String[], target_z = Float64[], dives_remain = Int[],dive_interval = Int[],surface_interval = Int[], cell = Int[])

    for i in 1:Nsp
        for j in 1:params[i,"Abundance"]
            id = "sp"*string(i)*"_ind"*string(j)
            spec = params[i,"SpeciesLong"]
            x0,y0,z0 = random_placement(g,params,i) #Need to revise function to allow for directed placements
            cell = grid_cell(grid_df,x0,y0,z0)

            initAge  = rand() * params[i,"Tmax"]  # init_age
            initSize = params[i,"VBG_LOO"] * (1 - exp(-params[i,"VBG_K"] * (initAge - params[i,"VBG_t0"])))
            biomass = params[i,"LWR_a"]*initSize^params[i,"LWR_b"]
            energy = biomass*0.2
            
            new_ind = Dict("ID" => id,"Species" => spec, "x" => x0, "y" => y0,"z" => z0, "length"=> initSize,"weight"=> biomass,"energy"=> energy,"age" => initAge,"generation"=> 1, "move" => "steady", "target_z" => z0, "dives_remain" => params[i,"Dive_Frequency"],"dive_interval" => params[i,"Dive_Interval"],"surface_interval" => params[i,"Surface_Interval"], "cell" => cell)

            push!(rawdata,new_ind)
        end
    end
    return rawdata
end



