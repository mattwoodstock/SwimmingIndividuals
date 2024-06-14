using BenchmarkTools
function update!(sim::MarineSimulation; time_offset = (vels = false, PARF = false, temp = false))
    @info "Starting model..."
    process_memory_before = Sys.total_memory() * 1e-9
    println(process_memory_before)
    for i in 1:sim.iterations
        TimeStep!(sim)
        process_memory_after = Sys.total_memory() * 1e-9
        println("Total memory used by the Julia process: $process_memory_after Gb")

    end
end