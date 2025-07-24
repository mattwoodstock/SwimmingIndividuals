# ===================================================================
# GPU-Compliant Visual Range Calculation
# ===================================================================

# Kernel to calculate the visual range for prey for each agent in parallel.
@kernel function visual_range_kernel!(vis_out, length, depth, min_prey, max_prey, surface_irradiance)
    ind = @index(Global)

    # Constants
    pred_contrast = 0.3f0
    salt = 30.0f0
    attenuation_coefficient = 0.32f0 - 0.016f0 * salt
    
    # Agent-specific calculations
    ind_length_m = length[ind] / 1000.0f0
    rmax = ind_length_m * 30.0f0
    
    # Simplified image size calculation for kernel
    pred_image = 0.75f0 * (ind_length_m / 0.01f0) * (ind_length_m / 4.0f0)
    eye_sensitivity = (rmax^2) / (pred_image * pred_contrast)
    
    # Light at depth
    I_z = surface_irradiance * exp(-attenuation_coefficient * depth[ind])
    
    # --- FIX: Calculate prey size factor based on actual prey size ---
    # The visual range is determined by the size of the target. We use the
    # smallest prey the predator will consider as the limiting factor.
    min_prey_length_m = ind_length_m * min_prey
    
    # Ensure prey size factor is not zero to avoid division by zero
    if min_prey_length_m > 0.0f0
        # The original formula for r_sq was physically incorrect. This is a more standard and stable formulation.
        # Reaction distance (r) is proportional to predator length and prey length.
        r_sq = (I_z * min_prey_length_m^2) / (eye_sensitivity * pred_contrast^2)
        r = r_sq > 0.0f0 ? sqrt(r_sq) : 0.0f0
        
        @inbounds vis_out[ind] = clamp(r, 0.0f0, rmax)
    else
        @inbounds vis_out[ind] = 0.0f0
    end
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
function behavior(model::MarineModel, sp::Int64, ind::Vector{Int32}, outputs::MarineOutputs)
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
        decision_inds = Int32.(intersect(ind, not_migrating_indices))
        
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

function prefixsum!(out::CuArray{Int}, input::CuArray{Int})
    result = CUDA.cumsum(input)         # Allocates new array
    copyto!(out, result)                # Copy back into the preallocated output
    return out
end

function find_foraging_indices!(
    time::CuArray{Float32},
    gut_fullness::CuArray{Float32},
    max_fullness::CuArray{Float32},
    inds::CuArray{T}
) where {T <: Integer}
    still_forage_mask = (time .> 0f0) .& (gut_fullness .< max_fullness)
    filtered_inds = inds[still_forage_mask]
    return filtered_inds
end

# Core decision-making function for foraging and movement.
function decision(model::MarineModel, sp::Int, ind::Vector{Int32}, outputs::MarineOutputs)
    sp_dat = model.individuals.animals[sp].data
    arch = model.arch

    # ---- Move relevant data to GPU once ----
    ind_gpu = CuArray(Int32.(ind))
    n = length(ind_gpu)

    gut_fullness_gpu = CuArray(Float32.(sp_dat.gut_fullness[ind]))
    biomass_gpu = CuArray(Float32.(sp_dat.biomass_school[ind]))
    max_fullness_gpu = 0.2f0 .* biomass_gpu
    feed_trigger_gpu = gut_fullness_gpu ./ max.(1.0f-9, max_fullness_gpu)

    rand_vals = CUDA.rand(Float32, n)
    eat_mask_gpu = feed_trigger_gpu .<= rand_vals
    eating_gpu = ind_gpu[eat_mask_gpu]  # Initial foragers

    # Allocate a full-length time array on GPU (indexed by full predator index)
    time_gpu = CUDA.fill(Float32(model.dt * 60.0), length(sp_dat.alive))

    # Preallocate buffers used in loop
    buf_gut = similar(gut_fullness_gpu)
    buf_biomass = similar(biomass_gpu)
    buf_max_fullness = similar(max_fullness_gpu)
    buf_time = similar(time_gpu, length(eating_gpu))
    
    # -------- Foraging Loop --------
    num_foraging_attempts = 5
    print("eat | ")
    for i in 1:num_foraging_attempts
        isempty(eating_gpu) && break

        # GPU: Identify prey
        calculate_distances_prey!(model, sp, eating_gpu)

        resolve_consumption!(model, sp, Array(eating_gpu))  # Only works on CPU-side indices

        # GPU: Apply consumption and update time
        apply_consumption!(model, sp, time_gpu, outputs)

        # Filter: keep only those who consumed something
        ration_vals = CuArray(Float32.(sp_dat.successful_ration[Array(eating_gpu)]))
        nonzero_mask = ration_vals .> 0f0
        eating_gpu = eating_gpu[nonzero_mask]  # Keep only individuals who ate something

        # Stop if none left
        isempty(eating_gpu) && break

        # Recalculate who should continue foraging
        gut_fullness_now = CuArray(Float32.(sp_dat.gut_fullness[Array(eating_gpu)]))
        biomass_now = CuArray(Float32.(sp_dat.biomass_school[Array(eating_gpu)]))
        max_fullness_now = 0.2f0 .* biomass_now
        time_now = CuArray(Float32.(time_gpu[eating_gpu]))

        eating_gpu = find_foraging_indices!(
            time_now, gut_fullness_now, max_fullness_now, eating_gpu
        )

    end
    print("move | ")

    # Movement based on remaining time
    movement_toward_habitat!(model, sp, time_gpu)

    return nothing
end

# ===================================================================
# CPU-based Initialization Functions (used only during setup)
# ===================================================================

function visual_range_preds_init(length,depth,min_pred,max_pred,ind)
    pred_contrast = fill(0.3,ind)
    salt = fill(30, ind)
    attenuation_coefficient = 0.32 .- 0.016 .* salt
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
    attenuation_coefficient = 0.32 .- 0.016 .* salt
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
