function update!(sim::MarineSimulation; time_offset = (vels = false, PARF = false, temp = false))
    #Run model
    start = now()
    println(start)
    for i in 1:sim.iterations
        TimeStep!(sim)

        #Open a text file to write the profile results
        #open("profile_output.txt", "w") do file
        #    Profile.print(file)
        #end
        #ProfileView.view()
    end
    stop = now()
    println(stop)
    println(stop-start)
end