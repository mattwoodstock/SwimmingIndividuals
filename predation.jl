function detection_distance(prey_length,pred_df,pred_ind)

    prey_length = prey_length * 1000
    salt = 30 #Needs to be in PSU
    surface_irradiance = 300 #Need real value. in W m-2
    prey_width = prey_length/4
    rmax = pred_df.data.length[pred_ind]*1000 # The maximum visual range. Currently this is 1 body length
    prey_contrast = 0.3 #Utne-Plam (1999)   
    eye_saturation = 1
    
    #Light attentuation coefficient
    a_lat = 0.64 - 0.016 * salt #Aksnes et al. 2009; per meter

    #Beam Attentuation coefficient
    c_lat = 4.87*a_lat #Huse and Fiksen (2010); per meter

    #Ambient irradiance at foraging depth
    i_td = surface_irradiance*exp(-0.1 * pred_df.data.z[pred_ind]) #Currently only reflects surface; 

    #Prey image area 
    prey_image = 0.75*prey_length*prey_width

    #Visual eye sensitivity
    eye_sensitivity = (rmax^2)/(prey_image*prey_contrast)
    
    #Equations for Visual Field
    f(x) = x^2 * exp(c_lat * x) - prey_contrast * prey_image * eye_sensitivity * i_td / (eye_saturation + i_td)
    fp(x) = 2*x * exp(c_lat * x) + c_lat * x^2 * exp(c_lat * x)

    x = NewtonRaphson(f,fp,pred_df.data.length[pred_ind])

    #Visual range estimates may be much more complicated when considering bioluminescent organisms. Could incorporate this if we assigned each species a "luminous type"
    ##https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3886326/
    return x
end

function holling_2(IBM_prey,pool_prey,model,pred_array,pred_spec,pred_ind,density,dt,outputs)
    #https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002392
    cmax = 0.1 * pred_array.data.weight[pred_ind] #Maximum daily ration = 10% of fishes body weight

    #Follows a Type 2 functional response, accounting for food previously consumed that day. 
    #encounter = density/(density+(cmax-pred_array.data.daily_ration[pred_ind]))
    encounter = density ./ (density .+ (cmax - pred_array.data.daily_ration[pred_ind]))

    capture_success = 0.7 #Likelihood a predator consumes individual targetted preys

    target_q = capture_success * encounter * (cmax-pred_array.data.daily_ration[pred_ind])

    pool_prop = 0.5 #Needs to be input, but is currently set here
    pool_time = dt*pool_prop #Amount of subtime spent foraging for pooled prey.
    t_left = dt #Reset time remaining

    if target_q > 0 #The predator can still eat
        while (t_left > dt-pool_time) & (pred_array.data.daily_ration[pred_ind] < target_q*(1-pool_prop)) & (nrow(IBM_prey) > 0)  #Forage for IBM, if they exist, there is time left, and they still can consume this much

            #move_time = move_predator!() #Move predator to new location
            chosen_prey = IBM_prey[IBM_prey.Distance .== minimum(IBM_prey.Distance),:] #Choose nearest prey. Will want to optimize with the prey_choice() function if prey selectivity is added.

            ## Add consumed biomass to food web
            outputs.foodweb.consumption[pred_spec,chosen_prey.Sp[1],Int(pred_array.data.pool_z[pred_ind]),model.iteration] += chosen_prey.Weight[1]

            #Move predator to prey
            move_time = move_predator!(pred_array,pred_spec,pred_ind,chosen_prey)

            pred_array.data.daily_ration[pred_ind] += IBM_prey.Weight[1] #Add prey weight to daily ration.

            #Fill predator gut
            pred_array.data.gut_fullness[pred_ind] = pred_array.data.gut_fullness[pred_ind] + (chosen_prey.Weight[1]/pred_array.data.weight[pred_ind])  

            #Remove consumed prey from model and add mortality
            remove_animal!(model,chosen_prey)
            #pred_mortality!(morts::Mortalities,chosen_prey) ##Need to write.

            #Delete individual from IBM_prey for next iteration.
            deleteat!(IBM_prey,findall(IBM_prey.Sp .== chosen_prey.Sp[1] .&& IBM_prey.Ind .== chosen_prey.Ind[1]))
            t_left -= move_time
        end

        ##Pool species
        pool_q = target_q*pool_prop #Amount consumed in pool

        ## Identification of minimum and maximum prey size for predator.
        #Prey limitation. Need to make species-specific
        min_prey_limit = 0.01 #Animals cannot eat anything less than 1% of their body length
        max_prey_limit = 0.05 #Animals cannot eat anything greater than 5% of their body length
        min_prey = pred_array.data.length[pred_ind] * min_prey_limit
        max_prey = pred_array.data.length[pred_ind] * max_prey_limit

        for i in 1:nrow(pool_prey)
            group_q = pool_q *pool_prey.Relative[pool_prey.Pool[i]]

            #Calculate the weight of an individual.
            ## May want to complicate this and use a size distribution. Otherwise, this creates a pretty simplistic view of plankton size distributions.
            name = Symbol("pool"*string(pool_prey.Pool[i]))
            group_array = getfield(model.pools.pool, name)

            ind_weight = group_array.characters.LWR_a[2][pool_prey.Pool[i]] * ((min_prey+max_prey)/2)^group_array.characters.LWR_b[2][pool_prey.Pool[i]] #Prey individuals are assumed to be the mean of the possible consumed preys

            outputs.foodweb.consumption[pred_spec,(model.n_species+pool_prey.Pool[i]),Int(pred_array.data.pool_z[pred_ind]),model.iteration] += ind_weight*group_q
            #Group consumption is scaled to the predator's remaining ration and the weight of each consumed individual
            pred_array.data.daily_ration[pred_ind] += ind_weight*group_q
        end
    end
    return nothing
end

function distance_matrix(lat1, lon1, depth1, lat2, lon2, depth2)
    R = 6371000  # Earth radius in meters
    
    #Uses the haversine formula to calculate x,y distance
    
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    
    a = sin(dlat / 2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon / 2)^2
    c = 2 * atan(sqrt(a), sqrt(1 - a))
    
    x = R * c

    y = depth2 - depth1

    distance = sqrt((y^2)+ (x^2))
    return distance
end

function calculate_distances(model::MarineModel)
    num_animals = sum(model.ninds)
    
    distances = zeros(Float64, num_animals, num_animals)
    count1 = count2 = 0

    for i in 1:model.n_species
        #Find number of individuals in species i
        name = Symbol("sp"*string(i))
        spec_array1 = getfield(model.individuals.animals, name)
        for j in 1:model.ninds[i]
            count1 += 1
            for k in 1:model.n_species
                #Find number of individuals in species k
                name2 = Symbol("sp"*string(k))
                spec_array2 = getfield(model.individuals.animals, name2)
                for l in 1:model.ninds[k]
                    count2 += 1
                    if (j != l) | (i != k)

                        distances[count1, count2] = distance_matrix(spec_array1.data.y[j], spec_array1.data.x[j], spec_array1.data.z[j], spec_array2.data.y[l], spec_array1.data.x[l], spec_array2.data.z[l])

                    end
                end
            end
            count2 = 0
        end
    end

    return distances
end

function available_prey(model::MarineModel,d_matrix,pred,pred_spec,pred_array,dt)
    
    #Prey limitation. Need to make species-specific
    min_prey_limit = 0.01 #Animals cannot eat anything less than 1% of their body length
    max_prey_limit = 0.05 #Animals cannot eat anything greater than 5% of their body length

    #Prey Detection Distances
    #https://onlinelibrary.wiley.com/doi/pdf/10.1111/geb.13782

    #Should this be restricted, or should it be assumed the fishes are constantly swimming?

    swim_velo = pred_array.p.Swim_velo[2][pred_spec] * pred_array.data.length[pred] /100 * 60 * dt

    prey_list = DataFrame(ID = Int[], Sp = Int[], Ind = Int[], x = Float64[], y = Float64[], z = Float64[], Weight = Float64[], Distance = Float64[])

    #Create dataframe for potential preys.
    prey_count = 1
    max_detection = 0 #Maximum detection range of the individual.

    for i in 1:model.n_species #Cycle through each potential prey species
        name = Symbol("sp"*string(i))
        spec_array1 = getfield(model.individuals.animals, name)

        for j in 1:model.ninds[i] #Cycle through each potential prey
            if (spec_array1.data.x[j] != -1) #Animal cannot eat itself, or already dead prey
                if (pred == j) & (pred_spec == i)
                    #Predator cannot eat itself
                else
                    prey_length = spec_array1.data.length[j] #Potential prey length

                    if (prey_length >= pred_array.data.length[pred] * min_prey_limit) & (prey_length <= pred_array.data.length[pred] * max_prey_limit)

                        prey_distance = d_matrix[pred,prey_count] #Distance to prey item

                        detection = detection_distance(prey_length,pred_array,pred)

                        if detection > max_detection
                            max_detection = detection
                        end

                        if (prey_distance <= detection)

                            new_row = Dict("ID" => prey_count, "Sp" => i, "Ind" => j, "x" => spec_array1.data.x[j], "y" => spec_array1.data.y[j], "z" => spec_array1.data.z[j],"Weight" => spec_array1.data.weight[j], "Distance" => prey_distance)
                            #Add individual to prey list
                            push!(prey_list,new_row)
                            prey_count += 1
                        end
                    end
                end
            end
        end
    end

    if model.dimension == 1
        searched_area = max_detection * 2 #Prey will be condensed. This includes visual range on z-axis.
    else
        searched_area = (4/3) * pi * max_detection^3 #Calculate the maximum searched sphere for the predator (i.e., maximum search volume)
    end
    return prey_list, searched_area
end

function move_predator!(pred_df,pred_spec,pred_ind,prey_df)

    #Handling time is a function of gut fullness with 2.0 seconds as the base. May want a better source.
    ##https://cdnsciencepub.com/doi/pdf/10.1139/f74-186
    ##seconds of Handling time from Langbehn et al. 2019. Essentially a cool-off period after feeding.

    handling_time_0 = 2.0

    handling_time = (1.19 - 1.24 * pred_df.data.gut_fullness[pred_ind] + 3.6 * pred_df.data.gut_fullness[pred_ind]^2 / handling_time_0) * handling_time_0

    #Identify x,y,z of prey

    pred_df.data.x[pred_ind] = prey_df.x[1]
    pred_df.data.y[pred_ind] = prey_df.y[1]
    #pred_df.data.z[pred_ind] = prey_df.z[1]

    #Calculate time to swim to prey
    swim_velo = pred_df.p.Swim_velo[2][pred_spec] * pred_df.data.length[pred_ind] /100

    time_to_prey = prey_df.Distance[1]/swim_velo

    t = ((handling_time + time_to_prey) / 60)

    #Add foraging time to activity time
    pred_df.data.active_time[pred_ind] = pred_df.data.active_time[pred_ind] + time_to_prey/60

    return t
end

function fill_gut!(pred_df,pred_ind,prey_df)

    prop_filled = prey_df.Weight[1]/pred_df.weight[pred_ind]

    pred_df.gut_fullness[pred_ind] = pred_df.gut_fullness[pred_ind] + prop_filled        

    #Delete this or explain purpose?
    #if pred_df.gut_fullness[pred_ind] > 0.03*pred_df.weight[pred_ind]
    #    pred_df.gut_fullness[pred_ind] = 0.03*pred_df.weight[pred_ind]
    #end
    return nothing
end

function prey_density(model,species_array,pred_ind,preys,area)
    #Calculate density of IBM preys
    IBM_dens = 0
    if size(preys,1) > 0 #There are preys within range
        IBM_dens = size(preys,1) / area #Number of preys per cubic meter of water
    end

    #Calculate density of pooled preys
    ##Add a size-selective component to this.
    #depthres = grid[grid.Name .== "depthres", :Value][1]
    #z_interval = maxdepth/depthres
    pool_dens = 0

    pool_list = DataFrame(Pool = Int[], Dens = Float64[])

    ## Identification of minimum and maximum prey size for predator.
    #Prey limitation. Need to make species-specific
    min_prey_limit = 0.01 #Animals cannot eat anything less than 1% of their body length
    max_prey_limit = 0.05 #Animals cannot eat anything greater than 5% of their body length
    min_prey = species_array.data.length[pred_ind] * min_prey_limit
    max_prey = species_array.data.length[pred_ind] * max_prey_limit

    for i in eachindex(fieldnames(typeof(model.pools.pool)))
        ## Should calculate the proportion of this density that is within the size range of the predator.
        name = Symbol("pool"*string(i))
        group_array = getfield(model.pools.pool, name)

        dens = group_array.density.num[Int(species_array.data.pool_z[pred_ind])]

        #Calculate a normal distribution of size classes
        #Calculate the proportion of inds in that size class that are within the appropriate size range.
        samples = sample_normal(group_array.characters.Min_Size[2][i],group_array.characters.Max_Size[2][i])
        prop_in_range = length(samples[(samples .>= min_prey) .& (samples .<= max_prey)])/length(samples)
 
        pool_dens += dens*prop_in_range #Add pooled density to density

        new_row = Dict("Pool" => i, "Dens" => dens*prop_in_range)
        push!(pool_list,new_row)
    end

    #Calculate relative abundances for weighted feeding.
    pool_list[:,"Relative"] .= pool_list.Dens / sum(pool_list.Dens)

    #Remove rows with only zero
    filter!(row -> row.Relative != 0,pool_list)

    #Sum IBM and pooled preys
    density = IBM_dens + pool_dens
    return density, pool_list
end


function eat!(model::MarineModel,d_matrix,i,j,spec_array1,dt,outputs)
    ddt = dt #Subset of time

    if (model.dimension == 1) #Running the 1D model

        #Calculate encounter range (i.e., distance an animal can perceive)
        ## Assumed prey length is 10% of predator

        IBM_prey, search_area = available_prey(model,d_matrix,j,i,spec_array1,dt)

        ### Need to calculate prey prey_densities by potential preys
        density, pool_prey = prey_density(model,spec_array1,j,IBM_prey,search_area) #Need to create function

        #prey length = mean length of available preys

        @profile holling_2(IBM_prey,pool_prey,model,spec_array1,i,j,density,dt,outputs)

        #profile_stats()
        filename = "Predation Profile.txt"
        f = open(filename, "w")
        Profile.print(f)
        close(f)

        ProfileView.view()

        throw(ErrorException("stop"))

        return nothing
    else #Running the 3D model

        ## Need to optimize with prey density function. Available prey function is fine because the difference between 1D and 3D is the presense of X,Y coords, but Z dimension still exists and distance calculation is the same.
        prey_list, search_area = available_prey(model,d_matrix,j,i,spec_array1,dt)

        #Replace following with prey density and q functions.
        
        while ddt > 0 ## Can only eat if there is time left

            if (nrow(prey_list) > 0) && (spec_array1.data.gut_fullness[j] < 1) # There are preys within range. Need to choose one and "remove" it.

                    chosen_prey = prey_choice(prey_list)
                    pred_success = rand()

                    if pred_success >= 0.3 #70% chance of predator success in a feeding event https://onlinelibrary.wiley.com/doi/pdf/10.1111/jfb.14451
                    remove_animal!(model,chosen_prey)
                    ddt = move_predator!(spec_array1,i,j,chosen_prey)
                    
                    fill_gut!(spec_array1.data,j,chosen_prey)
                    allocate_energy(spec_array1,i,j,chosen_prey)
                end
                    #Still remove animal from prey list as if it goes away
                    deleteat!(prey_list,findall(prey_list.Sp .== chosen_prey.Sp[1] .&& prey_list.Ind .== chosen_prey.Ind[1]))
            else
                chosen_prey = nothing
                ddt = 0 ## No preys within range, therefore we do not need this.
            end
        end
    end
end
