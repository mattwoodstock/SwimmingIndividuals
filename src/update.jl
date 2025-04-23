function update!(sim::MarineSimulation)
    #Run model
    for i in 1:sim.iterations
        TimeStep!(sim)
        #@profile TimeStep!(sim)

        #ProfileView.view()
        #stop
        #Open a text file to write the profile results
        #open("profile_output.txt", "w") do file
        #    Profile.print(file)
        #end
        
    end
end