function predation_mortality(model::MarineModel,df,outputs)
    if model.iteration > model.spinup
        model.individuals.animals[df.Sp[1]].data.x[df.Ind[1]] = 5e6
        model.individuals.animals[df.Sp[1]].data.y[df.Ind[1]] = 5e6
        model.individuals.animals[df.Sp[1]].data.z[df.Ind[1]] = 5e6
        model.individuals.animals[df.Sp[1]].data.ac[df.Ind[1]] = 0.0
        model.individuals.animals[df.Sp[1]].data.energy[df.Ind[1]] = -500
        model.individuals.animals[df.Sp[1]].data.behavior[df.Ind[1]] = 4
        model.individuals.animals[df.Sp[1]].data.biomass[df.Ind[1]] = 0
        outputs.mortalities[df.Sp[1],1] += 1 #Add one to the predation mortality column
        outputs.production[model.iteration,df.Sp[1]] += model.individuals.animals[df.Sp[1]].data.biomass[df.Ind[1]] #For P/B iteration
        #model.abund[df.Sp[1]] -=1
    end
    return nothing
end

function starvation(model,dead_sp, sp, i, outputs)
    if model.iteration > model.spinup
        dead_sp.data.x[i] .= 5e6
        dead_sp.data.y[i] .= 5e6
        dead_sp.data.z[i] .= 5e6
        dead_sp.data.ac[i] .= 0.0
        dead_sp.data.biomass[i] .= 0.0
        dead_sp.data.behavior[i] .= 4
        outputs.mortalities[sp,2] += 1 #Add one to the starvation mortality column
        #outputs.production[model.iteration,sp] .+= model.individuals.animals[sp].data.weight[i] #For P/B iteration
        #model.abund[sp] -=1
    end
    return nothing
end

function reduce_pool(model,pool,ind,ration)
    model.pools.pool[pool].data.biomass[ind] -= ration[1]

    if model.pools.pool[pool].data.biomass[ind] == 0 #Make sure pool actually stays alive
        model.pools.pool[pool].data.biomass[ind] = 1e-9
    end
    return nothing
end