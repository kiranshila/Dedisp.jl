using SIGPROC, Plots, Dedisp, BenchmarkTools, CUDA
pyplot()
default(; fmt=:png)

δt = 10e-6
n_samp = round(Int32,1/δt)
n_chan = 2048
f_min = 1280
f_max = 1530

t_total = n_samp * δt
dm_max = t_total / (Dedisp.KDM * (f_min^-2 - f_max^-2)) / 2
dm_min = 10
n_dm = 1024

fb = SIGPROC.fake_pulse(300, f_max, f_min; samples=n_samp, channels=n_chan, t_step=δt)
pulse = cu(fb.data.data)
freqs = cu(collect(fb.data.dims[2]))
dms = cu(collect(range(dm_min, dm_max; length=n_dm)))

# fb = Filterbank("/home/kiran/Downloads/candidate_ovro_20200428.fil")
# pulse = cu(fb.data.data)
# n_samp, n_chan = size(pulse)
# freqs = cu(collect(fb.data.dims[2]))
# dms = cu(collect(range(dm_min, dm_max; length=n_dm)))
# δt = step(fb.data.dims[1])
# f_min, f_max = extrema(freqs)

##### Non chunked
plan = plan_dedisp(freqs, f_max, dms, δt)
# Shifts at t0 will capture the longest dms
# So we'll look up to the DM minimum
output = CUDA.fill(Float32(mean(pulse)*n_chan),n_samp÷2, n_dm)
out = dedisp!(output, pulse, plan)

# Pretty Plot
heatmap((δt .* (1:(n_samp ÷ 2))), Array(dms), Array(standardize(out))'; clims=(0, 10),
        xlabel="Starting Time Offset (s)", ylabel="DM",c=:jet)

##### Chunked
#n_chunk = 128

#plan = plan_chunked_dedisp(freqs,f_max,dms,δt,n_chunk)
#tmp = CUDA.zeros(n_samp, n_dm, n_chunk)
#out = dedisp_in_chunks!(tmp,pulse, plan)