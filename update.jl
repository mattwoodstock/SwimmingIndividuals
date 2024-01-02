using ProgressBars
function update!(sim::MarineSimulation; time_offset = (vels = false, PARF = false, temp = false))
    #Will be used to change to environmental grids   
    #if sim.input.vels ≠ (;)
    #    for i in 1:sim.iterations
    #        model_t_vels = time_offset.vels ? (i-1)*sim.ΔT : sim.model.t
    #        model_t_PARF = time_offset.PARF ? (i-1)*sim.ΔT : sim.model.t
    #        model_t_temp = time_offset.temp ? (i-1)*sim.ΔT : sim.model.t
    #
    #        t_vel = floor(Int, model_t_vels/sim.input.ΔT_vel)+1 # starting from 1
    #        vel_copy!(sim.model.timestepper.vel₁, sim.input.vels.u[:,:,:,t_vel],
    #                sim.input.vels.v[:,:,:,t_vel], sim.input.vels.w[:,:,:,t_vel], sim.model.grid)
#
 #           t_par = floor(Int,model_t_PARF/sim.input.ΔT_PAR)+1 # starting from 1
  #          copyto!(sim.model.timestepper.PARF, sim.input.PARF[:,:,t_par])
#
 #           t_temp = floor(Int,model_t_temp/sim.input.ΔT_temp)+1 # starting from 1
   #         copy_interior!(sim.model.timestepper.temp, sim.input.temp[:,:,:,t_temp], sim.model.grid)
#
            #TimeStep!(sim.model, sim.ΔT)
#
            #write_output!(sim.output_writer, sim.model, sim.ΔT)
    #    end
   # else
        for i in 1:sim.iterations

            #model_t_PARF = time_offset.PARF ? sim.model.t : (i-1)*sim.ΔT
            #model_t_temp = time_offset.temp ? sim.model.t : (i-1)*sim.ΔT
            #t_par = floor(Int,model_t_PARF/sim.input.ΔT_PAR)+1 # starting from 1
            #copyto!(sim.model.timestepper.PARF, sim.input.PARF[:,:,t_par])
            #t_temp = floor(Int,model_t_temp/sim.input.ΔT_temp)+1 # starting from 1
            #copy_interior!(sim.model.timestepper.temp, sim.input.temp[:,:,:,t_temp], sim.model.grid)
            
            TimeStep!(sim.model, sim.ΔT,sim.temp,sim.outputs)
            #Timestep-specific outputs
            #depth_density(sim.model,i,sim.depth_dens)
            #write_output!(sim.output_writer, sim.model, sim.ΔT)
        end
    #end
end