function shed_eDNA!(model,species_df,sp,ind)
    #Find index of first eDNA particle that is currently inactive
    index = findfirst(x -> x == -1, model.parts.eDNA.data.x)

    #Add particle to ecosystem at x,y,z of animal
    model.parts.eDNA.data.x[index] = species_df.data.x[ind]
    model.parts.eDNA.data.y[index] = species_df.data.y[ind]
    model.parts.eDNA.data.z[index] = species_df.data.z[ind]
    model.parts.eDNA.data.lifespan[index] = model.parts.eDNA.state.Decay_rate[2][1]

end

function decay_eDNA!()
    indices = findall(x -> x < 0, model.parts.eDNA.data.lifespan)
    model.parts.eDNA.data.x[indices] .= -1
    model.parts.eDNA.data.y[indices] .= -1
    model.parts.eDNA.data.z[indices] .= -1
end