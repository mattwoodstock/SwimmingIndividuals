function remove_animal!(model::MarineModel, df)

    name = Symbol("sp"*string(df.Sp[1]))
    spec_array1 = getfield(model.individuals.animals, name)

    spec_array1.data.x[df.Ind[1]] = -1
    spec_array1.data.y[df.Ind[1]] = -1
end