# ===================================================================
# GPU-Compliant Visual Range Calculation
# ===================================================================

# Kernel to calculate the visual range for prey for each agent in parallel.
@kernel function visual_range_kernel!(vis_out, length, depth, min_prey, max_prey, surface_irradiance)
    ind = @index(Global)

    # Constants
    pred_contrast = 0.3
    salt = 30.0
    attenuation_coefficient = 0.64 - 0.016 * salt
    
    # Agent-specific calculations
    ind_length_m = length[ind] / 1000.0
    rmax = ind_length_m * 30.0
    
    # Simplified image size calculation for kernel
    pred_image = 0.75 * (ind_length_m / 0.01) * (ind_length_m / 4.0)
    eye_sensitivity = (rmax^2) / (pred_image * pred_contrast)
    
    # Light at depth
    I_z = surface_irradiance * exp(-attenuation_coefficient * depth[ind])
    
    prey_size_factor = (min_prey + max_prey) / 2.0
    
    #  visual range calculation
    r_sq = (I_z * ind_length_m^2) / (pred_contrast * eye_sensitivity * prey_size_factor)
    r = r_sq > 0 ? sqrt(r_sq) : 0.0
    
    @inbounds vis_out[ind] = clamp(r, 0.0, rmax)
end

# Launcher for the visual range kernel.
function update_visual_range!(model::MarineModel, sp::Int)
    arch = model.arch
    data = model.individuals.animals[sp].data
    p = model.individuals.animals[sp].p
    
    surface_irradiance = ipar_curve(model.t)
    
    kernel! = visual_range_kernel!(device(arch), 256, (length(data.x),))
    kernel!(
        data.vis_prey, data.length, data.z,
        p.Min_Prey[2][sp], p.Max_Prey[2][sp],
        surface_irradiance
    )
    KernelAbstractions.synchronize(device(arch))
end

# ===================================================================
# Core Behavioral Functions
# ===================================================================

# Main behavioral dispatcher.
function behavior(model::MarineModel, sp::Int, ind::Vector{Int}, outputs::MarineOutputs)
    behave_type = model.individuals.animals[sp].p.Type[2][sp]
    sp_dat = model.individuals.animals[sp].data
    
    # Update visual range for all agents of this species first
    update_visual_range!(model, sp)

    if behave_type == "dvm_strong"
        dvm_action!(model, sp) # This kernel runs on all agents
        
        # Find which agents are NOT migrating to pass them to the decision function
        mig_status_cpu = Array(sp_dat.mig_status)
        not_migrating_indices = findall(x -> x <= 0.0, mig_status_cpu)
        
        # Intersect with the list of currently living agents
        decision_inds = intersect(ind, not_migrating_indices)
        
        if !isempty(decision_inds)
            decision(model, sp, decision_inds, outputs)
        end

    elseif behave_type == "dvm_weak"
        # This logic needs to be converted to a kernel for full GPU compliance.
        # For now, we assume it runs on CPU or has been refactored.
        dvm_action!(model, sp)
        decision(model, sp, ind, outputs)

    elseif behave_type == "pelagic_diver"
        dive_action!(model, sp)
        decision(model, sp, ind, outputs)
        
    elseif behave_type == "non_mig"
        decision(model, sp, ind, outputs)
    end
    
    return nothing
end


# Core decision-making function for foraging and movement.
function decision(model::MarineModel, sp::Int, ind::Vector{Int64}, outputs::MarineOutputs)
    sp_dat = model.individuals.animals[sp].data
    arch = model.arch

    # Create views to work with the subset of individuals
    gut_fullness_view = @view sp_dat.gut_fullness[ind]
    biomass_school_view = @view sp_dat.biomass_school[ind]

    max_fullness = 0.2 .* biomass_school_view
    feed_trigger = gut_fullness_view ./ max_fullness
    
    # Generate random numbers on the correct device
    local val1
    if arch isa GPU
        val1 = CUDA.rand(Float32, length(ind))
    else
        val1 = rand(Float32, length(ind))
    end

    # Create a boolean mask on the GPU, copy it to CPU, then findall
    eat_mask_gpu = (feed_trigger .<= val1)
    eat_mask_cpu = Array(eat_mask_gpu)
    to_eat_indices = findall(eat_mask_cpu)

    eating = ind[to_eat_indices]
    
    # Create the 'time' vector on the correct device
    time = array_type(arch)(zeros(Float32, length(sp_dat.alive)))
    time .= Float32(model.dt * 60.0)

    if !isempty(eating)
        # --- Predation Sequence ---
        print("find prey | ")
        calculate_distances_prey!(model, sp, eating)
        
        resolve_consumption!(model, sp, eating)
        
        print("eat | ")
        apply_consumption!(model, sp, time,outputs)
    end

    print("move | ")
    movement_toward_habitat!(model, sp, time)
    
    return nothing
end


# ===================================================================
# CPU-based Initialization Functions (used only during setup)
# ===================================================================

function visual_range_preds_init(length,depth,min_pred,max_pred,ind)
    pred_contrast = fill(0.3,ind)
    salt = fill(30, ind)
    attenuation_coefficient = 0.64 .- 0.016 .* salt
    ind_length = length ./ 1000
    pred_length = ind_length ./ 0.01
    pred_width = pred_length ./ 4
    pred_image = 0.75 .* pred_length .* pred_width
    rmax = ind_length .* 30
    eye_sensitivity = (rmax.^2) ./ (pred_image .* pred_contrast)
    surface_irradiance = ipar_curve(0)
    I_z = surface_irradiance .* exp.(-attenuation_coefficient .* depth)
    pred_size_factor = 1+((min_pred+max_pred)/2)
    r = max.(1,ind_length .* sqrt.(I_z ./ (pred_contrast .* eye_sensitivity .* pred_size_factor)))
    return r
end

function visual_range_preys_init(length,depth,min_prey,max_prey,ind)
    pred_contrast = fill(0.3,ind)
    salt = fill(30, ind)
    attenuation_coefficient = 0.64 .- 0.016 .* salt
    ind_length = length ./ 1000
    pred_length = ind_length ./ 0.01
    pred_width = pred_length ./ 4
    pred_image = 0.75 .* pred_length .* pred_width
    rmax = ind_length .* 30
    eye_sensitivity = (rmax.^2) ./ (pred_image .* pred_contrast)
    surface_irradiance = ipar_curve(0)
    I_z = surface_irradiance .* exp.(-attenuation_coefficient .* depth)
    prey_size_factor = (min_prey+max_prey)/2
    r = min.(1,ind_length .* sqrt.(I_z ./ (pred_contrast .* eye_sensitivity .* prey_size_factor)))
    return r
end
