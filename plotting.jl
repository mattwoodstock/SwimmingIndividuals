function reshape_for_heatmap(df)
    dat = Matrix{Int}(undef,length(unique(df.Depth))-1,length(unique(df.Time))-1)
    count = 1
    for i in 2:length(unique(df.Time))
        for j in 2:length(unique(df.Depth))
            dat[j-1,i-1] = df.Number[count]
            count += 1
        end
    end

    x = unique(df.Time)[2:length(unique(df.Time))]
    y = unique(df.Depth)[2:length(unique(df.Depth))]
    return x,y,dat
end


function food_web_plot(output,model,dt)

    plotdir = joinpath(pwd(),"Plots")

    # Generate random trophic levels for each species (between 1 and 5). Will inform with calculation later.
    trophic_levels = rand(model.n_species) * 5

    # Generate random x-axis positions to avoid overlap. Set seed for this later.
    x_positions = rand(model.n_species)


    for time in 1:length(output.foodweb.consumption[1,1,1,:])
        t = time * dt #Calculate real time in minutes
        if t % 60 == 0 #Make plot on the hour
            for z in 1:length(output.foodweb.consumption[1,1,:,1])
                # Wrap this all in a loop later
                consumption_matrix = output.foodweb.consumption[1:model.n_species,1:model.n_species,z,time]



                # Biomass for node size. Wrap this later as well.
                biomass_values = output.foodweb.biomasses[1:model.n_species,z,time]

                # Set the size of the plot (width, height) in inches
                #plot_size = (4, 3)

                hour = t/60
                # Initialize the plot
                plot(
                    nodesize = biomass_values .* 20,  # Node size scaled to biomass
                    nodecolor = :lightblue,
                    xlims = (0, 1),
                    ylims = (minimum(trophic_levels)-0.5, maximum(trophic_levels)+0.5),  # Adjust the y-axis limits based on your trophic level range
                    title = "Food Web: Hour $hour | Depth Bin $z"
                )


                # Add nodes to the plot
                scatter!(x_positions, trophic_levels, markersize=biomass_values .* 100, color=:lightblue, legend=false)

                arrow_size = consumption_matrix * 5000

                # Draw line segments between nodes based on trophic levels and biomass_matrix
                for i in 1:model.n_species
                    for j in 1:model.n_species
                        if consumption_matrix[j,i] > 0
                            plot!([x_positions[j], x_positions[i]], [trophic_levels[j], trophic_levels[i]], color=:black, line=(:arrow,arrow_size[j,i],0.4,0.1))
                        end
                    end
                end

                display(plot!(legend=false))

                # Show the plot
                savefig(joinpath(plotdir,"Plot Hour $hour _Depth $z.png"))
            end
        end
    end
end