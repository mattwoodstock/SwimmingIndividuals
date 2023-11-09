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

function dive_initiate(z_t,target_z,trait,rate)
    status = "descend"
    target_z = (rand() * (trait.Dive_depth_max[1] - trait.Dive_depth_min[1]))+trait.Dive_depth_min[1]
    z_t = z_t + rate
    return z_t, target_z, status
end

function surface_initiate(z_t,target_z,trait,rate)
    status = "ascend"
    target_z = (rand() * (trait.Night_depth_max[1] - trait.Night_depth_min[1]))+trait.Night_depth_min[1]
    z_t = z_t - rate
    return z_t, target_z, status
end

function descend_dvm_start(z_t,target_z,trait,vm_rate)
    status = "descend"
    target_z = (rand() * (trait.Day_depth_max[1] - trait.Day_depth_min[1]))+trait.Day_depth_min[1]
    z_t = z_t + vm_rate
    return z_t, target_z, status
end

function ascend_dvm_start(z_t,target_z,trait,vm_rate)
    status = "ascend"
    target_z = (rand() * (trait.Night_depth_max[1] - trait.Night_depth_min[1]))+trait.Night_depth_min[1]
    z_t = z_t - vm_rate
    return z_t, target_z, status
end

function vertical_movement(ind,trait,g,t)
    mig = ind.move[1]
    vm_rate = trait.Mig_rate[1] #m per minute (Bianchi et al. (2016))
    dvm_trigger = 0.1 #proportion of bodyweight energy necessary for asynchronous migrators to migrate
    z_t = ind.z[1]
    target_z = ind.target_z[1]
    dive_interval = ind.dive_interval[1]
    surface_interval = ind.surface_interval[1]

    ## Movement modes: steady, ascend, descend, dive_int, surface_int, spent
    if trait.Movement_type[1] == "Epipelagic diver"
        if (mig == "steady") && (ind.dives_remain[1] > 0)
            if rand() > 0.95 #5% chance of diving each minute
                dive_rate = ind.length[1]/100 * trait.Swim_velo[1] * 60
                z_t, target_z, mig = dive_initiate(z_t,target_z,trait,dive_rate)
            end
        elseif mig == "descend"
            surface_interval = trait.surface_interval[1] #Reset surface interval
            if z_t >=target_z #Reached target descend depth
                mig = "dive_int"
            else
                z_t = z_t + dive_rate
            end
        elseif mig == "dive_int"
            if dive_interval == 0
                dive_rate = ind.length[1]/100 * trait.Swim_velo[1] * 60
                z_t, target_z, mig = surface_initiate(z_t,target_z,trait,dive_rate)
            else
                dive_interval = dive_interval - 1
            end
        elseif mig == "ascend"
            dive_interval = trait.dive_interval[1] #Reset dive interval
            if z_t <= target_z
                mig = "surface_int"
            else
                z_t = z_t - dive_rate
            end
        elseif mig == "surface_int"
            if surface_interval == 0
                mig = "steady"
            else
                surface_interval = surface_interval - 1
            end
        end
    elseif trait.Movement_type[1] == "Surface associated diver"
        #Same as epipelagic diver, but animal returns to the surface

        if (mig == "steady") && (ind.dives_remain[1] > 0)
            if rand() > 0.95 #5% chance of diving each minute
                dive_rate = ind.length[1]/100 * trait.Swim_velo[1] * 60
                z_t, target_z, mig = dive_initiate(z_t,target_z,trait,dive_rate)
            end
        elseif mig == "descend"
            surface_interval = trait.surface_interval[1] #Reset surface interval
            if z_t >=target_z #Reached target descend depth
                mig = "dive_int"
            else
                z_t = z_t + dive_rate
            end
        elseif mig == "dive_int"
            if dive_interval == 0
                dive_rate = ind.length[1]/100 * trait.Swim_velo[1] * 60
                z_t, target_z, mig = surface_initiate(z_t,target_z,trait,dive_rate)
                target_z = 0 #All surface-associated animals go to surface
            else
                dive_interval = dive_interval - 1
            end
        elseif mig == "ascend"
            dive_interval = trait.dive_interval[1] #Reset dive interval
            if z_t <= target_z
                mig = "surface_int"
            else
                z_t = z_t - dive_rate
            end
        elseif mig == "surface_int"
            if surface_interval == 0
                mig = "steady"
            else
                surface_interval = surface_interval - 1
            end
        end    
    elseif trait.Movement_type[1] == "Synchronous vertical migrator"
        if (t >= 6*60) && (mig == "steady") && (t < 18*60)
            mig = "descend"
            target_z = (rand() * (trait.Day_depth_max[1] - trait.Day_depth_min[1]))+trait.Day_depth_min[1]
            z_t = z_t + vm_rate
        end
        if  (t >= 18*60) && (mig == "spent")
            mig == "ascend"
            target_z = (rand() * (trait.Night_depth_max[1] - trait.Night_depth_min[1]))+trait.Night_depth_min[1]
            z_t = z_t - vm_rate
        end
        if (mig == "ascend") && (z_t <= target_z)
            mig == "steady"
        else
            z_t = z_t - vm_rate
        end
        if (mig == "descend") && (z_t >= target_z)
            mig == "spent"
        else
            z_t = z_t + vm_rate
        end     
    elseif trait.Movement_type[1] == "Asynchronous vertical migrator"
        if (t >= 6*60) && (mig == "steady") && (t < 18*60)
            mig = "descend"
            target_z = (rand() * (trait.Day_depth_max[1] - trait.Day_depth_min[1]))+trait.Day_depth_min[1]
            z_t = z_t + vm_rate
        end
        if  (t >= 18*60) && (mig == "spent") && (energy < weight*dvm_trigger)
            mig == "ascend"
            target_z = (rand() * (trait.Night_depth_max[1] - trait.Night_depth_min[1]))+trait.Night_depth_min[1]
            z_t = z_t - vm_rate
        end
        if (mig == "ascend") && (z_t <= target_z)
            mig == "steady"
        else
            z_t = z_t - vm_rate
        end
        if (mig == "descend") && (z_t >= target_z)
            mig == "spent"
        else
            z_t = z_t + vm_rate
        end     
    end
    return z_t, mig, target_z, dive_interval, surface_interval
end
