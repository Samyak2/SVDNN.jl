using LinearAlgebra
using Flux
using Flux: glorot_uniform
using Zygote: Params

using TimerOutputs
const to = TimerOutput()

# below implementation is based on original Dense implementation https://github.com/FluxML/Flux.jl/blob/7a32a703f0f2842dda73d4454aff5990ade365d5/src/layers/basic.jl#L85-L104

"""
    SVDense(in::Integer, out::Integer, σ = identity)
Create a SVD-based `Dense` layer with parameters `W1`, `W2` and `b`.
    y = σ.(W1 * (W2 * x) .+ b)
The input `x` must be a vector of length `in`, or a batch of vectors represented
as an `in × N` matrix. The out `y` will be a vector or batch of length `out`.
# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))

"""
struct SVDense{F,S,T}
  W1::S
  W2::S
  b::T
  σ::F
end

SVDense(W1, W2, b) = SVDense(W1, W2, b, identity)

function SVDense(m::Integer, n::Integer, l::Integer, σ = identity;
               initW = glorot_uniform, initb = zeros)
  return SVDense(initW(n, l), initW(l, m), initb(n), σ)
end

function TruncatedSVD(A, l)
    F = svd(A)
    return F.U[:, 1:l], Diagonal(F.S)[1:l, 1:l], F.Vt[1:l, :]
end

function SVDense(model::Dense, l::Integer)
  U, s, Vt = TruncatedSVD(model.W, l)
  W1 = U
  W2 = s*Vt
  return SVDense(W1, W2, model.b, model.σ)
end


Flux.@functor SVDense

function (a::SVDense)(x::AbstractArray)
  W1, W2, b, σ = a.W1, a.W2, a.b, a.σ
  σ.(W1*(W2*x) .+ b)
end

function Base.show(io::IO, l::SVDense)
  print(io, "SVDense(", size(l.W2, 2), ", ", size(l.W1, 1), ", l=", size(l.W1, 2))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

# Try to avoid hitting generic matmul in some simple cases
# Base's matmul is so slow that it's worth the extra conversion to hit BLAS
(a::SVDense{<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::SVDense{<:Any,W})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

# END of code from https://github.com/FluxML/Flux.jl/blob/7a32a703f0f2842dda73d4454aff5990ade365d5/src/layers/basic.jl#L85-L104

