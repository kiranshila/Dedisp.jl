using LoopVectorization

circmod(x,y) = mod(x-1,y) + 1

function dedisp(source::AbstractMatrix{T}, plan) where {T <: Real}
    n_samp, n_chan = size(source)
    _, n_dm = size(plan)
    output = zeros(T, n_samp, n_dm)
    μ = mean(source)
    @tturbo for i in 1:n_dm
        for k in 1:n_samp
            for j in 1:n_chan
                shifted_samp_idx = circmod(k + plan[j,i],n_samp)
                output[k, i] += source[shifted_samp_idx, j] / n_samp
            end
        end
    end
    return output
end
