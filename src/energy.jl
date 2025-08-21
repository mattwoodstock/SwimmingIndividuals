function energy!(model::MarineModel, sp::Int, temp::AbstractArray, indices, outputs,current_date)
    arch = model.arch
    p_cpu = model.individuals.animals[sp].p
    dt = model.dt

    # --- 1. GATHER DATA ---
    agent_data_device = model.individuals.animals[sp].data

    # FIX: Create a new StructArray on the CPU by copying each GPU array individually
    data_cpu = StructArray(
        unique_id = Array(agent_data_device.unique_id),x = Array(agent_data_device.x), y = Array(agent_data_device.y), z = Array(agent_data_device.z),
        length = Array(agent_data_device.length), abundance = Array(agent_data_device.abundance),
        biomass_ind = Array(agent_data_device.biomass_ind), biomass_school = Array(agent_data_device.biomass_school),
        energy = Array(agent_data_device.energy), gut_fullness = Array(agent_data_device.gut_fullness),
        cost = Array(agent_data_device.cost), pool_x = Array(agent_data_device.pool_x),
        pool_y = Array(agent_data_device.pool_y), pool_z = Array(agent_data_device.pool_z),
        active = Array(agent_data_device.active), ration = Array(agent_data_device.ration),
        alive = Array(agent_data_device.alive), vis_prey = Array(agent_data_device.vis_prey),
        target_pool_x = Array(agent_data_device.target_pool_x),
        target_pool_y = Array(agent_data_device.target_pool_y),
        mature = Array(agent_data_device.mature), age = Array(agent_data_device.age),
        cell_id = Array(agent_data_device.cell_id), sorted_id = Array(agent_data_device.sorted_id),
        repro_energy = Array(agent_data_device.repro_energy),
        best_prey_dist = Array(agent_data_device.best_prey_dist),
        best_prey_idx = Array(agent_data_device.best_prey_idx),
        best_prey_sp = Array(agent_data_device.best_prey_sp),
        best_prey_type = Array(agent_data_device.best_prey_type),
        successful_ration = Array(agent_data_device.successful_ration),
        temp_idx = Array(agent_data_device.temp_idx),
        cell_starts = Array(agent_data_device.cell_starts),
        cell_ends = Array(agent_data_device.cell_ends),
        mig_status = Array(agent_data_device.mig_status),
        target_z = Array(agent_data_device.target_z),
        interval = Array(agent_data_device.interval),
        generation = Array(agent_data_device.generation)
    )
    
    daily_births = model.daily_birth_counters[sp]
    temp_cpu = Array(temp)
    size_bin_thresholds_cpu = Array(model.size_bin_thresholds)
    smort_cpu = Array(outputs.Smort)

    # --- 2. COMPUTE BIOENERGETICS ON CPU ---
    spawn_season = CSV.read(model.files[model.files.File .== "reproduction", :Destination][1], DataFrame)
    species_name = p_cpu.SpeciesLong.second[sp]
    row_idx = findfirst(==(species_name), spawn_season.Species)
    spawn_val = row_idx !== nothing ? spawn_season[row_idx, model.environment.ts + 1] : 0.0
    
    spinup_check = model.iteration > model.spinup

    for ind in 1:length(data_cpu.x)
        if (data_cpu.alive[ind] == 1.0) && (data_cpu.age[ind] > p_cpu.Larval_Duration.second[sp])

            # Gather Agent Properties
            my_temp = temp_cpu[ind]; my_consumed = data_cpu.ration[ind]
            my_weight_ind = data_cpu.biomass_ind[ind]
            my_active_time = data_cpu.active[ind] / dt
            my_abundance = data_cpu.abundance[ind]
            my_z = data_cpu.z[ind]; my_length = data_cpu.length[ind]
            my_energy = data_cpu.energy[ind]; my_mature = data_cpu.mature[ind]

            # Get Species-Specific Trait Codes
            energy_ed = p_cpu.Energy_density.second[sp]; taxa_code = p_cpu.Taxa.second[sp]
            energy_type_code = p_cpu.MR_type.second[sp]; max_size = p_cpu.Max_Size.second[sp]

            activity_multiplier = 2.5 # Default value that could be adjusted

            # --- Respiration (R) ---
            R = 0.0
            oxy_joules_per_mg = 13.6
            if my_weight_ind > 0 && my_abundance > 0
                if energy_type_code == 2 # "cetacean"
                    joules_per_kcal = 4184.0
                    # Calculate Field Metabolic Rate (FMR) in kcal/day
                    min_fmr_kcal_day = 350.0 * (my_weight_ind / 1000.0)^0.75
                    max_fmr_kcal_day = 420.0 * (my_weight_ind / 1000.0)^0.75
                    # Convert to Joules per timestep
                    min_fmr_J_ts = min_fmr_kcal_day * joules_per_kcal * (dt / 1440.0)
                    max_fmr_J_ts = max_fmr_kcal_day * joules_per_kcal * (dt / 1440.0)
                    R_ind = min_fmr_J_ts + (max_fmr_J_ts - min_fmr_J_ts) * my_active_time
                elseif energy_type_code == 3 # "deepsea"
                    depth = max(1.0, my_z)
                    log_weight = log(my_weight_ind); inv_temp = 1000.0 / (273.15 + my_temp); log_depth = log(depth)
                    lnr = (taxa_code == 1 ? 19.491 + 0.885*log_weight - 5.770*inv_temp - 0.261*log_depth :
                           taxa_code == 2 ? 28.326 + 0.779*log_weight - 7.903*inv_temp - 0.365*log_depth :
                           18.775 + 0.766*log_weight - 5.265*inv_temp - 0.113*log_depth)
                    rate_mg_per_kg_per_hr = exp(lnr); my_weight_kg = my_weight_ind / 1000.0
                    rate_mg_per_ind_per_hr = rate_mg_per_kg_per_hr * my_weight_kg
                    rate_J_per_hr = rate_mg_per_ind_per_hr * oxy_joules_per_mg
                    R_ind = rate_J_per_hr * (dt / 60.0)
                else # Default
                    resp_a = 0.05   # Intercept (mg O2/g/day)
                    resp_b = -0.2   # Allometric scaling exponent for mass-specific rate
                    resp_q = 0.069  # Temperature sensitivity (Q10)

                    smr_mass_specific = resp_a * my_weight_ind^resp_b
                    
                    smr_temp_corrected = smr_mass_specific * exp(resp_q * my_temp)
                    
                    smr_total_daily = smr_temp_corrected * my_weight_ind
                    smr_joules_per_day = smr_total_daily * oxy_joules_per_mg
                    smr_per_timestep = smr_joules_per_day * (dt / 1440.0)
                    
                    cost_resting = smr_per_timestep * (1.0 - my_active_time)
                    cost_active = (smr_per_timestep * activity_multiplier) * my_active_time
                    R_ind = cost_resting + cost_active
                end
                R = R_ind * my_abundance
            end

            # --- Net Energy and Gut Evacuation ---
            sda_coeff, egestion_coeff, excretion_coeff = 0.05, 0.1, 0.05
            total_waste_and_cost = R + (my_consumed * (sda_coeff + egestion_coeff + excretion_coeff))
            net_energy = my_consumed - total_waste_and_cost
            if ismissing(total_waste_and_cost); continue; end
            data_cpu.cost[ind] = total_waste_and_cost
            my_energy += net_energy

            evac_prop = min(1.0, 0.053 * exp(0.073 * my_temp))
            if evac_prop < 1.0
                data_cpu.gut_fullness[ind] *= exp((dt / 60.0) * log(1.0 - evac_prop))
            else
                data_cpu.gut_fullness[ind] = 0.0
            end

            # --- Growth & Reproduction ---
            lwr_a = p_cpu.LWR_a.second[sp]; lwr_b = p_cpu.LWR_b.second[sp]
            energy_reserve_coeff = 0.2

            r_max = my_weight_ind * my_abundance * energy_ed * energy_reserve_coeff
            excess = max(0.0, my_energy - r_max)
            my_energy = min(r_max, my_energy)

            if !spinup_check
                excess = 0.0
            end

            if my_length < max_size && excess > 0.0
                growth_prop = exp(-5.0 * my_length / max_size)
                growth_energy = excess * growth_prop

                growth_biomass_ind = (growth_energy / my_abundance) / energy_ed
                
                # 2. Add this growth to the individual's biomass
                new_biomass_ind = my_weight_ind + growth_biomass_ind
                
                # 3. Calculate the new length from the new individual biomass
                new_length = 10.0 * (new_biomass_ind / lwr_a)^(1.0 / lwr_b)

                if new_length < 0 
                    println(growth_energy)
                    println(growth_biomass_ind)
                end
                
                # 4. Update the agent's state variables
                data_cpu.biomass_ind[ind] = new_biomass_ind
                data_cpu.length[ind] = new_length
                data_cpu.biomass_school[ind] += (growth_energy / energy_ed)
                
                my_energy -= growth_energy
                excess -= growth_energy
            end

            if spinup_check && my_mature == 1.0 && excess > 0.0 && spawn_val > 0.0
                data_cpu.repro_energy[ind] = excess
                my_energy -= excess
            end

            data_cpu.energy[ind] = my_energy
            
            # --- Starvation ---
            if my_energy < 0.0 && spinup_check
                data_cpu.alive[ind] = 0.0

                # Get the agent's grid coordinates
                x = data_cpu.pool_x[ind]
                y = data_cpu.pool_y[ind]
                z = data_cpu.pool_z[ind]

                # Use the CPU version of the thresholds array
                size_bin = find_species_size_bin(data_cpu.length[ind], sp, size_bin_thresholds_cpu)

                # Update the CPU version of the Smort array
                if size_bin > 0
                    smort_cpu[x, y, z, sp, size_bin] += my_weight_ind
                end
            end

            # --- Senescence ---
            if spinup_check

                # Calculate senescence probability
                senescence_prob = exp(50.0 * (my_length / max_size - 0.95))
                
                # Check if the agent dies from old age
                if rand(Float32) < senescence_prob
                    data_cpu.alive[ind] = 0.0
                end
            end
        end
    end

    ## Double check that larvae cannot lose energy
    larval_mask = (data_cpu.alive .== 1.0) .& (data_cpu.age .<= p_cpu.Larval_Duration.second[sp])
    if any(larval_mask)
        data_cpu.cost[larval_mask] .= 0.0f0
    end
    
    # --- Process Reproduction on the CPU ---
    repro_inds = findall((data_cpu.repro_energy .> 0) .& (data_cpu.alive .== 1.0) .& (data_cpu.age .> p_cpu.Larval_Duration.second[sp]))

    if !isempty(repro_inds) && spinup_check && spawn_val > 0
        parent_data_for_repro = (
            x = data_cpu.x[repro_inds], y = data_cpu.y[repro_inds], z = data_cpu.z[repro_inds],
            pool_x = data_cpu.pool_x[repro_inds], pool_y = data_cpu.pool_y[repro_inds], pool_z = data_cpu.pool_z[repro_inds],
            biomass_ind = data_cpu.biomass_ind[repro_inds], generation = data_cpu.generation[repro_inds]
        )
        
        repro_result = calculate_new_offspring_cpu(p_cpu, parent_data_for_repro, data_cpu.repro_energy[repro_inds], spawn_val, sp, current_date, daily_births)
        
        if repro_result !== nothing
            new_offspring, daily_births_update = repro_result
            model.daily_birth_counters[sp] = daily_births_update
            
            num_new = length(new_offspring.x)

            if num_new > 0
                dead_slots = findall(data_cpu.alive .== 0)
                
                # This block now correctly handles resizing when needed
                if num_new > length(dead_slots)
                    current_maxN = length(data_cpu.x)
                    needed_slots = num_new - length(dead_slots)
                    new_maxN = max(floor(Int, current_maxN * 1.5), current_maxN + needed_slots)
                    
                    copyto!(model.individuals.animals[sp].data, data_cpu)
                    copyto!(outputs.Smort, smort_cpu)

                    resize_agent_storage!(model, sp, new_maxN)

                    agent_data_device = model.individuals.animals[sp].data
                    data_cpu = StructArray(NamedTuple(k => Array(v) for (k, v) in pairs(StructArrays.components(agent_data_device))))
                    smort_cpu = Array(outputs.Smort)
                    
                    dead_slots = findall(data_cpu.alive .== 0)
                end

                num_to_add = min(num_new, length(dead_slots))

                if num_to_add > 0
                    slots_to_fill = @view dead_slots[1:num_to_add]
                    for (field, values) in pairs(new_offspring)
                        getproperty(data_cpu, field)[slots_to_fill] .= @view(values[1:num_to_add])
                    end
                    data_cpu.alive[slots_to_fill] .= 1.0
                end
            end
        end
        data_cpu.repro_energy[repro_inds] .= 0.0
    end

    # --- 3. UPDATE: Copy the modified CPU data back to the original device arrays ---
    copyto!(outputs.Smort, smort_cpu)
    copyto!(model.individuals.animals[sp].data, data_cpu)

    return nothing
end