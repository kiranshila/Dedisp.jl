using CUDA, Statistics, LLVM, LLVM.Interop

function dedisp_kernel!(output::AbstractMatrix{T}, source, plan, μ) where {T}
    # Throw out all of our safety garuntees
    assume.(size(source) .> 0)
    assume.(size(output) .> 0)
    assume.(size(plan) .> 0)

    # Pull out constants
    n_samp, n_chan = size(source)
    samp_idx = (blockIdx().y - Int32(1)) * blockDim().x + threadIdx().x
    dm_idx = blockIdx().x
    n_out_samp, _ = size(output)

    # Bail early if we get an oob sample index
    if samp_idx > n_out_samp
        return nothing
    end

    # Copy shifts into shared memory in a parallel fashion
    shifts = @cuDynamicSharedMem(UInt32, n_chan)
    @inbounds for chan_idx in (threadIdx().x):(blockDim().x):n_chan
        shifts[chan_idx] = plan[chan_idx, dm_idx]
    end
    sync_threads()

    # Sum
    x = zero(T)
    @inbounds for chan_idx in Int32(1):n_chan
        shifted_samp_idx = samp_idx + shifts[chan_idx]
        if shifted_samp_idx > n_samp
            x += μ
        else
            x += source[shifted_samp_idx, chan_idx]
        end
    end

    @inbounds output[samp_idx, dm_idx] = x
    # Kernel is side-effecting, doesn't return
    return nothing
end

"""
    dedisp!(output,source,plan)

Dedisperses dynamic spectra `source` in-place into `output` according to the dedispersion plan `plan`.
- `source` has dimensions time samples x frequency channels
- `output` has dimensions starting time samples x DM trials
- `plan` has dimensions frequency channels x DM trials
"""
function dedisp!(output::CuArray{A,2}, source, plan::CuArray{C,2}) where {A,C}
    _, n_chan = size(source)
    _, n_dm = size(plan)
    n_out_samp, _ = size(output)

    # Precompute mean
    μ = Float32(mean(source))

    # Compile kernel and grab capabilities
    kernel = @cuda launch = false dedisp_kernel!(output, source, plan, μ)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    blocks = (n_dm, cld(n_out_samp, threads))

    # Run kernel
    kernel(output, source, plan, μ; threads=threads, blocks=blocks,
           shmem=sizeof(UInt32) * n_chan)

    # Normalize
    return output ./ n_chan
end

# Source is n_samp * n_chan
# output is n_samp * n_dm * n_chunk
# plan is n_chan_chunk * n_dm * n_chunk
# slices is n_chan_chunk * n_chunk
function dedisp_chunks_kernel!(output::AbstractArray{T,3}, source, plan, μ) where {T}
    # Throw out all of our safety garuntees
    assume.(size(source) .> 0)
    assume.(size(output) .> 0)
    assume.(size(plan) .> 0)

    # Pull out constants
    n_samp, _ = size(source)
    n_chan_chunk, _, _ = size(plan)

    samp_idx = (blockIdx().y - 1) * blockDim().x + threadIdx().x
    dm_idx = blockIdx().x
    chunk_idx = blockIdx().z

    # We need to figure out this thread's channel chunk in the source array
    chan_idxs = ((chunk_idx - 1) * n_chan_chunk + 1):(chunk_idx * n_chan_chunk)

    # We schedule more threads than there are samples to get to a multiple of 32
    if samp_idx > n_samp
        return nothing
    end

    # Copy shifts into dynamic memory in a parallel fashion
    shifts = @cuDynamicSharedMem(UInt32, n_chan_chunk)
    @inbounds for chan_idx in (threadIdx().x):(blockDim().x):n_chan_chunk
        shifts[chan_idx] = plan[chan_idx, dm_idx, chunk_idx]
    end
    sync_threads()

    # Sum
    x = zero(T)
    @inbounds for (shift_idx, chan_idx) in enumerate(chan_idxs)
        shifted_samp_idx = samp_idx + shifts[shift_idx]
        if shifted_samp_idx > n_samp
            x += μ
        else
            x += source[shifted_samp_idx, chan_idx]
        end
    end
    output[samp_idx, dm_idx, chunk_idx] = x

    # Kernel is side-effecting, doesn't return
    return nothing
end

"""
    plan_chunked_dedisp(freqs, f_max, dms, δt, n_chunk)

(GPU-Only) Identical to `plan_dedisp` except chunks the frequencies by `n_chunk`.
"""
function plan_chunked_dedisp(freqs, f_max, dms, δt, n_chunk)
    n_chan = length(freqs)
    subbands = cu(reshape(collect(1:n_chan), :, n_chunk))
    return Δt.(reshape(freqs[subbands], :, 1, n_chunk), f_max, dms', δt)
end

"""
    dedisp!(output, source, chunked_plan)

Identical to `dedisp!` except operates on chunked plans. Returned is the reduced output.
"""
function dedisp!(output::CuArray{A,3}, source,
                 plan::CuArray{C,3}) where {A,C}
    # Preallocate memory
    _, n_dm, n_chunks = size(plan)
    n_samp, n_chan = size(source)

    μ = A(mean(source))

    # Build kernel
    kernel = @cuda launch = false dedisp_chunks_kernel!(output, source, plan, μ)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    blocks = (n_dm, cld(n_samp, threads), n_chunks)

    # Call for each chunk
    kernel(output, source, plan, μ; threads=threads, blocks=blocks,
           shmem=sizeof(UInt32) * n_chan)

    # Wait for all the subband chunks to finish and reduce
    return sum(output; dims=3)
end

export dedisp!, plan_chunked_dedisp
