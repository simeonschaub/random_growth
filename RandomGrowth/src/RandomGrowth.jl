module RandomGrowth

using LoopVectorization
using Base.Threads
using Base.Experimental: @sync

function corner_growth_kernel!(A, n, m)
    for j in 2:m+1
        for i in 2:n+1
            @inbounds A[i, j] += max(A[i-1, j], A[i, j-1])
        end
    end
end

function _corner_growth!(W, kernel_size)
    N = cld(size(W, 1) - 1, kernel_size)
    M = cld(size(W, 2) - 1, kernel_size)
    for d in 1:M+N-1
        @threads for J in (d ≤ N ? 1 : d-N+1):min(d, M)
            I = d - J + 1

            i₀ = (I-1) * kernel_size + 1
            j₀ = (J-1) * kernel_size + 1

            i_range = i₀:min(i₀ + kernel_size, size(W, 1))
            j_range = j₀:min(j₀ + kernel_size, size(W, 2))
            #@show i_range, j_range

            kernel = view(W, i_range, j_range)
            corner_growth_kernel!(kernel, length(i_range)-1, length(j_range)-1)
        end
    end
end

function _corner_growth2!(W, kernel_size)
    N = cld(size(W, 1) - 1, kernel_size)
    M = cld(size(W, 2) - 1, kernel_size)
    for d in 1:M+N-1
        @sync for J in (d ≤ N ? 1 : d-N+1):min(d, M)
            I = d - J + 1

            i₀ = (I-1) * kernel_size + 1
            j₀ = (J-1) * kernel_size + 1

            i_range = i₀:min(i₀ + kernel_size, size(W, 1))
            j_range = j₀:min(j₀ + kernel_size, size(W, 2))

            kernel = view(W, i_range, j_range)
            @spawn corner_growth_kernel!(kernel, length(i_range)-1, length(j_range)-1)
        end
    end
end

function corner_growth!(W, kernel_size)
    W[1, 1] = 0
    @sync begin
        v1 = @view W[2:end, 1]
        @spawn accumulate!(+, $v1, $v1)
        v2 = @view W[1, 2:end]
        @spawn accumulate!(+, $v2, $v2)
    end
    _corner_growth!(W, kernel_size)
    return W
end

end
