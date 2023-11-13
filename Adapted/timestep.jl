function TimeStep!(model::MarineModel, ΔT, diags::PlanktonDiagnostics)
    # model.t = model.t+ΔT
    model.t = model.iteration * ΔT 

    #Vertical movement of animals
        
    end