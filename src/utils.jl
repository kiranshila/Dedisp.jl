"""
    standardize(A)

Transform matrix `A` into units of signal to noise ratio along optional dimension `dim`, defaulting to the first dimension.
"""
function standardize(A;dims=1)
    μ = mean(A; dims=dims)
    σ = std(A; mean=μ, dims=dims)
    return @. (A - μ) / σ
end

"""
    Δt(f_min, f_max, DM)

Given two frequency bounds, `f_min` and `f_max` in MHz, get the time shift in seconds for a dispersed pulse of dispersion measure `DM`.
"""
function Δt(f_min, f_max, DM)
    return KDM * DM * (f_min^-2 - f_max^-2)
end

"""
    Δt(f_min, f_max, DM, δt)

Given two frequency bounds, `f_min` and `f_max` in MHz, get the time shift in samples for a dispersed pulse of dispersion measure `DM`.
"""
function Δt(f_min, f_max, DM, δt)
    return round(Int32, Δt(f_min, f_max, DM) / δt)
end

"""
    plan_dedisp(freqs, f_max, dms, δt)

Create a dedispersion plan for that covers pulses dispersed over `freqs` with a maximum of `f_max` in MHz over all possible DMs in `dms` with source time resolution `δt`
"""
plan_dedisp(freqs, f_max, dms, δt) = Δt.(freqs, f_max, dms', δt)