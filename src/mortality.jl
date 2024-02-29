function predation_mortality(model::MarineModel,df,outputs)
    model.individuals.animals[df.Sp[1]].data.x[df.Ind[1]] = -1
    model.individuals.animals[df.Sp[1]].data.y[df.Ind[1]] = -1
    model.individuals.animals[df.Sp[1]].data.z[df.Ind[1]] = -1
    outputs.mortalities[df.Sp[1],1] += 1 #Add one to the predation mortality column
    outputs.production[model.iteration,df.Sp[1]] += model.individuals.animals[df.Sp[1]].data.weight[df.Ind[1]] #For P/B iteration
    return nothing
end

function starvation(dead_sp, sp, i, outputs)
    dead_sp.data.x[i] = -1
    dead_sp.data.y[i] = -1
    outputs.mortalities[sp,2] += 1 #Add one to the starvation mortality column
    outputs.production[model.iteration,sp] += model.individuals.animals[sp].data.weight[i] #For P/B iteration
    return nothing
end

function reduce_pool(model,pool,depth)
    cell_volume = 100 * 1 * 1 #Cell volume to remove
    model.pools.pool[pool].density.num[depth,] -= 1 / cell_volume
    if model.pools.pool[pool].density.num[depth,] < 0 #Find a better way to implement this?
        model.pools.pool[pool].density.num[depth,] = 0
    end
    return nothing
end