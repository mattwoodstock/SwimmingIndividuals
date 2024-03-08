function update!(sim::MarineSimulation; time_offset = (vels = false, PARF = false, temp = false))
    @info "Starting model..."
    for i in 1:sim.iterations
        TimeStep!(sim)
    end
end