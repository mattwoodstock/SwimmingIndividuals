mutable struct Diagnostics
    plankton::NamedTuple       # for each species
    tracer::NamedTuple         # for tracers
    iteration_interval::Int64  # time interval that the diagnostics is time averaged
end


function MarineDiagnostics(model; tracer=(),
    plankton=(:num, :graz, :mort, :dvid),
    iteration_interval::Int64 = 1)

@assert isa(tracer, Tuple)
@assert isa(plankton, Tuple)

diag_avail(tracer, plankton)

ntr   = length(tracer)
nproc = length(plankton)
trs   = []
procs = []

total_size = (model.grid.Nx+model.grid.Hx*2, model.grid.Ny+model.grid.Hy*2, model.grid.Nz+model.grid.Hz*2)

for i in 1:ntr
tr = zeros(total_size) |> array_type(model.arch)
push!(trs, tr)
end
tr_d1 = zeros(total_size) |> array_type(model.arch)
tr_d2 = zeros(total_size) |> array_type(model.arch)
tr_default = (PAR = tr_d1, T = tr_d2)

diag_tr = NamedTuple{tracer}(trs)
diag_tr = merge(diag_tr, tr_default) # add PAR as default diagnostic

plank_name = keys(model.individuals.phytos)
Nsp = length(plank_name)

for j in 1:Nsp
procs_sp = []
for k in 1:nproc
proc = zeros(total_size) |> array_type(model.arch)
push!(procs_sp, proc)
end
diag_proc = NamedTuple{plankton}(procs_sp)

procs_sp_d = []
for l in 1:4
proc = zeros(total_size) |> array_type(model.arch)
push!(procs_sp_d, proc)
end
diag_proc_default = NamedTuple{(:num, :graz, :mort, :dvid)}(procs_sp_d)

diag_proc = merge(diag_proc, diag_proc_default) # add num, graz, mort, and dvid as default diagnostics

push!(procs, diag_proc)
end
diag_sp = NamedTuple{plank_name}(procs)

diagnostics = Diagnostics(diag_sp, diag_tr, iteration_interval)

return diagnostics
end


function diag_avail(tracer, plank)
    tracer_avail = tracer_avail_diags()
    plank_avail  = plank_avail_diags()
    for i in 1:length(tracer)
        if length(findall(x->x==tracer[i], tracer_avail)) == 0
            throw(ArgumentError("$(tracer[i]) is not one of the diagnostics"))
        end
    end

    for i in 1:length(plank)
        if length(findall(x->x==plank[i], plank_avail)) == 0
            throw(ArgumentError("$(plank[i]) is not one of the diagnostics"))
        end
    end
end

function tracer_avail_diags()
    return (:PAR, :DIC, :DOC, :POC, :NH4, :NO3, :DON, :PON, :PO4, :DOP, :POP)
end

function plank_avail_diags()
    plank_avail = (:num, :graz, :mort, :dvid, :PS, :resp, :Bm, :Chl, :Th)
    return plank_avail
end