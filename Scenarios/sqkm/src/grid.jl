function random_placement(g_file::String,params,sp)
    
    grid = CSV.read(g_file,DataFrame)

    x0 = y0 = z0 = 0.0
    x0 = rand()
    y0 = rand()
    z0 = rand()

    x₁ = grid[grid.Name.=="lonmin",:Value][1]
    x₂ = grid[grid.Name.=="lonmax",:Value][1]
    xdiff = x₂ - x₁

    y₁ = grid[grid.Name.=="latmin",:Value][1]
    y₂ = grid[grid.Name.=="latmax",:Value][1]
    ydiff = y₂ - y₁

    z₁ = grid[grid.Name.=="depthmax",:Value][1]

    x0 = (x0 * xdiff) + x₁ # Longitude
    y0 = (y0 * ydiff) + y₁ # Latitude



    z0 = z0 * (params.Night_depth_max[sp] - params.Night_depth_min[sp])+params.Night_depth_min[sp] # All indiividuals start at their nighttime depth

    return x0, y0, z0
end

# Function to calculate the volume of a grid cell
function grid_cell_volume(lat1_deg, lon1_deg, lat2_deg, lon2_deg, depth_m)
    # Convert latitudes and longitudes to radians
    lat1_rad = lat1_deg * π / 180
    lon1_rad = lon1_deg * π / 180
    lat2_rad = lat2_deg * π / 180
    lon2_rad = lon2_deg * π / 180

    # Calculate distances between grid points along the Earth's surface
    R = 6371000  # Radius of the Earth in meters
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2)^2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)^2
    c = 2 * atan(sqrt(a))
    distance_m = R * c

    # Calculate the volume of the rectangular box
    width_m = distance_m
    length_m = distance_m
    volume_m3 = width_m * length_m * depth_m

    return volume_m3
end



function grid_dataframe(grid::AbstractGrid,g_file::String)
    
    g = CSV.read(g_file,DataFrame)
    gridFrame = DataFrame(cell = Int[],x=Float64[],y=Float64[],z=Float64[],Volume = Float64[])

    x₁ = g[g.Name.=="lonmin",:Value][1]
    y₁ = g[g.Name.=="latmin",:Value][1]
    z₁ = g[g.Name.=="depthmax",:Value][1]
    zdiff = z₁/g[g.Name.=="depthres",:Value][1]

    count = 1
    for i in 1:grid.Nx
        for j in 1:grid.Ny
            for k in 1:grid.Nz
                cell = count
                x = x₁+grid.Δx*i
                y = y₁+grid.Δy*j
                z = zdiff*k

                volume = grid_cell_volume(y,x,y+grid.Δy*j,x+grid.Δx*i,zdiff)

                new_cell = Dict("cell" => cell, "x" => x, "y" => y, "z" => z, "Volume" => volume)
                push!(gridFrame,new_cell)
                count = count + 1
            end
        end
    end
    return gridFrame
end

function grid_cell(g_frame,x1,y1,z1)
    filtered_grid1 = g_frame[(g_frame.x .> x1), :]

    if nrow(filtered_grid1) == 0
        filtered_grid = g_frame[(g_frame.x .<= x1),:]
    else
        filtered_grid = g_frame[(g_frame.x .> x1), :]
    end
    filtered_grid2 = filtered_grid[(filtered_grid.y .> y1), :]
    if nrow(filtered_grid2) == 0
        filtered_grid = filtered_grid[(filtered_grid.y .<= y1),:]
    else
        filtered_grid = filtered_grid[(filtered_grid.y .> y1), :]
    end
    filtered_grid3 = filtered_grid[(filtered_grid.z .> z1), :]
    if nrow(filtered_grid3) == 0
        filtered_grid = filtered_grid[(filtered_grid.z .<= z1),:]
    else
        filtered_grid = filtered_grid[(filtered_grid.z .> z1), :]
    end

    cell = filtered_grid.cell[1]
    return cell
end



