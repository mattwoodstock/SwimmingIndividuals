using Random, CSV

function horizontal_movement(ind,trait,g)
    grid = CSV.read("grid.csv",DataFrame)
    # Constants for Earth's radius and conversion factors
    R_earth = 6371000.0  # Earth's radius in meters
    degrees_to_radians = π / 180.0  # Conversion factor for degrees to radians

    # Calculate the distance the fish will travel
    distance = ind.length[1]/100 * trait.Swim_velo[1] * 60

    # Generate a random heading angle in degrees
    heading_angle_degrees = rand() * 360.0  # Random angle between 0 and 360 degrees

    # Convert the heading angle to radians
    heading_angle_radians = heading_angle_degrees * degrees_to_radians

    # Calculate the change in latitude and longitude based on the heading angle
    delta_lat = (distance / R_earth) * cos(heading_angle_radians)
    delta_lon = (distance / R_earth) * sin(heading_angle_radians)

    # Calculate the new latitude and longitude
    new_latitude = ind.y[1] + (delta_lat / degrees_to_radians)
    new_longitude = ind.x[1] + (delta_lon / degrees_to_radians)


    #Assure animal stays within the grid

    while (new_latitude < grid[grid.Name .== "latmin", :Value][1] || new_latitude > grid[grid.Name .== "latmax", :Value][1] || new_longitude < grid[grid.Name .== "lonmin", :Value][1] || new_longitude > grid[grid.Name .== "lonmax", :Value][1])
        # Generate a random heading angle in degrees
        heading_angle_degrees = rand() * 360.0  # Random angle between 0 and 360 degrees

        # Convert the heading angle to radians
        heading_angle_radians = heading_angle_degrees * degrees_to_radians

        # Calculate the change in latitude and longitude based on the heading angle
        delta_lat = (distance / R_earth) * cos(heading_angle_radians)
        delta_lon = (distance / R_earth) * sin(heading_angle_radians)

        # Calculate the new latitude and longitude
        new_latitude = ind.y[1] + (delta_lat / degrees_to_radians)
        new_longitude = ind.x[1] + (delta_lon / degrees_to_radians)
    end

    return (new_longitude,new_latitude)
end


function dvm_action(df,sp,t,ΔT)
    #All animals will go through this at each time step.
    #The DVM Trigger for synchronous migrators will be 1 (always migrate)
    #The DVM Trigger for asynchronous migrators will be normal
    #The DVM Rate for all non-migrators will be 0

    #Steady == 0, Ascending == 1, Descending == 2, Spent == -1

    #Migration speed = function of migration length Weibe et al. 2023
    #https://www.sciencedirect.com/science/article/pii/S0967063722001996


    dvm_trigger = df.p.DVM_trigger[2][sp] #proportion of bodyweight energy necessary for asynchronous migrators to migrate. Synchronous migrators will be 0 because they always migrate
    for ind in 1:length(df.data.length) #Loop through each individual

        t_adjust = 0.0 #Need to adjust the movement time if a migrator reaches the target depth in the middle of the time.

        if (t >= 6*60) && (df.data.mig_status[ind] == 0) && (t < 18*60) #Animal needs to start descending


            df.data.mig_status[ind] = 2
            df.data.target_z[ind] = rand(df.p.Day_depth_min[2][sp]:df.p.Day_depth_max[2][sp])

            df.data.mig_rate[ind] = 4.2 #Calculate DVM rate per minute

            t_adjust = minimum([ΔT,abs((df.data.target_z[ind] - df.data.z[ind]) / df.data.mig_rate[ind])]) #Time adjuster so that if animal reaches the target before times up, it is reflected in energetics

            df.data.z[ind] = minimum([df.data.target_z[ind],df.data.z[ind] + (df.data.mig_rate[ind] * ΔT)])



            df.data.active_time[ind] = df.data.active_time[ind] + t_adjust


            if df.data.z[ind] == df.data.target_z[ind]
                df.data.mig_status[ind] = -1
                df.data.feeding[ind] = 0 #Animal stops feeding
            end

        elseif  (t >= 18*60) && (df.data.mig_status[ind] == -1) && (df.data.energy[ind] < df.data.weight[ind]*dvm_trigger) #Animal needs to start ascending

            df.data.mig_status[ind] = 1

            df.data.target_z[ind] = rand(df.p.Night_depth_min[2][sp]:df.p.Night_depth_max[2][sp])


            df.data.mig_rate[ind] = 4.2 #Calculate DVM rate per minute

            df.data.z[ind] = maximum([df.data.target_z[ind],df.data.z[ind] - (df.data.mig_rate[ind] * ΔT)])

            if df.data.z[ind] == df.data.target_z[ind]
                df.data.mig_status[ind] = 0
                df.data.feeding[ind] = 1 #Animal starts feeding
            end

            t_adjust = minimum([ΔT,abs((df.data.target_z[ind] - df.data.z[ind]) / df.data.mig_rate[ind])]) #Time adjuster so that if animal reaches the target before times up, it is reflected in energetics

            df.data.active_time[ind] = df.data.active_time[ind] + t_adjust

        elseif (df.data.mig_status[ind] == 1) #Animal keeps ascending

            df.data.z[ind] = maximum([df.data.target_z[ind],df.data.z[ind] - (df.data.mig_rate[ind] * ΔT)])
            
            if df.data.z[ind] == df.data.target_z[ind]
                df.data.mig_status[ind] = 0
                df.data.feeding[ind] = 1 #Animal starts feeding
            end

            t_adjust = minimum([ΔT,abs((df.data.target_z[ind] - df.data.z[ind]) / df.data.mig_rate[ind])]) #Time adjuster so that if animal reaches the target before times up, it is reflected in energetics

            df.data.active_time[ind] = df.data.active_time[ind] + t_adjust

        elseif (df.data.mig_status[ind] == 2) #Animal keeps descending
            df.data.z[ind] = minimum([df.data.target_z[ind],df.data.z[ind] + (df.data.mig_rate[ind] * ΔT)])

            if df.data.z[ind] == df.data.target_z[ind]
                df.data.mig_status[ind] = -1
            end

            t_adjust = minimum([ΔT,abs((df.data.target_z[ind] - df.data.z[ind]) / df.data.mig_rate[ind])]) #Time adjuster so that if animal reaches the target before times up, it is reflected in energetics

            df.data.active_time[ind] = df.data.active_time[ind] + t_adjust

        end  
    end
    return nothing
end

function dive_action(df,sp,ΔT)
    #Surface Interval == 0, Diving == 1, Ascending == 2, Dive Interval == -1

    for ind in 1:length(df.data.length)
        if (df.data.dives_remaining[ind] > 0) && (df.data.interval[ind] <= 0) && (df.data.mig_status[ind] == 0) #Time to Dive 
            df.data.mig_status[ind] == 1
            df.data.target_z[ind] = rand(df.p.Dive_depth_min[2][sp]:df.p.Dive_depth_max[2][sp])
            df.data.dives_remaining[ind] = df.data.dives_remaining[ind] - 1

            df.data.mig_rate[ind] = (df.p.Swim_velo[2][sp]*df.data.length[ind]/100*60)

            df.data.z[ind] = minimum([df.data.target_z[ind],(df.data.z[ind]+(df.data.mig_rate[ind]*ΔT))])

            t_adjust = minimum([ΔT,abs((df.data.target_z[ind] - df.data.z[ind]) / df.data.mig_rate[ind])]) #Time adjuster so that if animal reaches the target before times up, it is reflected in energetics

            df.data.active_time[ind] = df.data.active_time[ind] + t_adjust

            if df.data.z[ind] == df.data.target_z[ind] #Animal reaches depth
                df.data.mig_status[ind] == -1
                df.data.interval[ind] = df.p.Dive_Interval[2][sp]
            end

        elseif (df.data.mig_status == 1) && (df.data.z[ind] <= df.data.target_z[ind]) #Animal is still diving

            df.data.z[ind] = minimum([df.data.target_z[ind],df.data.z[ind] + df.data.mig_rate[ind]*ΔT])

            t_adjust = minimum([ΔT:abs((df.data.target_z[ind] - df.data.z[ind]) / df.data.mig_rate[ind])]) #Time adjuster so that if animal reaches the target before times up, it is reflected in energetics

            df.data.active_time[ind] = df.data.active_time[ind] + t_adjust

            if df.data.z[ind] >= df.data.target_z[ind] #Animal reaches depth
                df.data.mig_status == -1
                df.data.interval[ind] = df.p.Dive_Interval[2][sp]

            end
        elseif df.data.mig_status[ind] == -1 ## In dive interval
            df.data.interval[ind] = df.data.interval[ind] - ΔT

            if df.data.interval[ind] <= 0 #End dive interval
                df.data.mig_status[ind] == 2
                df.data.target_z[ind] = 0 #Return to surface (Mammals only)

                df.data.mig_rate[ind] = (df.p.Swim_velo[2][sp]*df.data.length[ind]/100*60)
            
                df.data.z[ind] = maximum([df.data.target_z[ind],df.data.z[ind] - df.data.mig_rate[ind]*ΔT])

                t_adjust = minimum([ΔT:abs((df.data.target_z[ind] - df.data.z[ind]) / df.data.mig_rate[ind])]) #Time adjuster so that if animal reaches the target before times up, it is reflected in energetics

                df.data.active_time[ind] = df.data.active_time[ind] + t_adjust

                if df.data.z[ind] <= df.data.target_z[ind] #Animal reaches surface
                df.data.mig_status[ind] == 0
                df.data.interval[ind] = df.p.Surface_Interval[2][sp]
                end
            end
        elseif (df.data.mig_status[ind] == 0) || (df.data.dives_remaining[ind] == 0) ## In surface interval
            df.data.interval[ind] = df.data.interval[ind] - ΔT #If no dives are remaining, this will just count down till the end of the day, when it is reset.
        end
    end
    return nothing
end