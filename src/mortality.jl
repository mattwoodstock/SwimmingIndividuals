function predation_mortality(model::MarineModel,df,outputs,n_consumed,ration)
    if model.iteration > model.spinup
        model.individuals.animals[df.Sp[1]].data.abundance[df.Ind[1]] -= n_consumed
        model.individuals.animals[df.Sp[1]].data.biomass_school[df.Ind[1]] -= ration
        outputs.mortalities[df.Sp[1],1] += n_consumed #Add one to the predation mortality column

        if (model.individuals.animals[df.Sp[1]].data.abundance[df.Ind[1]]) == 0
            model.individuals.animals[df.Sp[1]].data.alive[df.Ind[1]] = 0.0
        end
    end
    return nothing
end

function resource_mortality(model)
    dt = model.dt
    for i in 1:model.n_resource
        z = model.resource_trait[i,:Z]
        z_conv = z/(365*1440) * dt
        P_removed = 1 - exp(-z_conv * dt)

        matching_idxs = findall(r -> r.sp == i,model.resources)
        for idx in matching_idxs
            model.resources[idx].biomass *= P_removed
        end
    end
end