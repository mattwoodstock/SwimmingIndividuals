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

    dvm_trigger = df.p.DVM_trigger[2][sp] #proportion of bodyweight energy necessary for asynchronous migrators to migrate. Synchronous migrators will be 0 because they always migrate
    for ind in 1:length(df.data.length) #Loop through each individual

        if (t >= 6*60) && (df.data.mig_status[ind] == 0) && (t < 18*60)
            df.data.mig_status[ind] = 2
            df.data.target_z[ind] = (rand() * (df.p.Day_depth_max[2][sp] - df.p.Day_depth_min[2][sp])) + df.p.Day_depth_min[2][sp]
            df.data.z[ind] = df.data.z[ind] + (df.p.Mig_rate[2][sp] * ΔT)
        end

        if  (t >= 18*60) && (df.data.mig_status[ind] == -1) && (df.data.energy[ind] < df.data.weight[ind]*dvm_trigger)

            df.data.mig_status[ind] = 1
            df.data.target_z[ind] = (rand() * (df.p.Night_depth_max[2][sp] - df.p.Night_depth_min[2][sp])) + df.p.Night_depth_min[2][sp]
            df.data.z[ind] = df.data.z[ind] - (df.p.Mig_rate[2][sp] * ΔT)
        end

        if (df.data.mig_status[ind] == 1) && (df.data.z[ind] <= df.data.target_z[ind])

            df.data.mig_status[ind] = 0
        elseif (df.data.mig_status[ind] == 1)

            df.data.z[ind] = df.data.z[ind] - (df.p.Mig_rate[2][sp] * ΔT)
        end

        if (df.data.mig_status[ind] == 2) && (df.data.z[ind] >= df.data.target_z[ind])

            df.data.mig_status[ind] = -1
        elseif (df.data.mig_status[ind] == 2)
            df.data.z[ind] = df.data.z[ind] + (df.p.Mig_rate[2][sp] * ΔT)
        end  
    end

    return df
end
