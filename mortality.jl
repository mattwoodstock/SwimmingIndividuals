function remove_animal!(model::MarineModel, df)

    name = Symbol("sp"*string(df.Sp[1]))
    spec_array1 = getfield(model.individuals.animals, name)

    spec_array1.data.x[df.Ind[1]] = -1
    spec_array1.data.y[df.Ind[1]] = -1
end

function starvation!(dead_sp, i)
    dead_sp.data.x[i] = -1
    dead_sp.data.y[i] = -1
end