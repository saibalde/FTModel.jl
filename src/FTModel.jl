module FTModel

using Printf
using LinearAlgebra
using Distributions
using Jacobi

import Base.show, Base.+, Base.-, Base.*, Base.zero, Base.rand, Base.sign

export FTCoeff, sumfuncoeffs, evaluate, gradient

mutable struct FTCoeff
    ndim::Int
    rank::Array{Int,1}
    nbasis::Array{Int,1}
    coeffs::Array{Array{Float64,3},1}

    function FTCoeff(ndim::Int, rank::Array{Int,1}, nbasis::Array{Int,1},
                     coeffs::Array{Array{Float64,3},1})
        if ndim <= 1
            error("Dimensionality must be at least two")
        end

        if length(rank) != ndim + 1
            error("Rank specification does not match dimensionality")
        end

        if rank[1] != 1 || rank[ndim + 1] != 1
            error("Boundary ranks must be one")
        end

        if length(nbasis) != ndim
            error("Basis expansion specification does not match dimensionality")
        end

        for d in 1:ndim
            if size(coeffs[d], 1) != rank[d] ||
               size(coeffs[d], 2) != rank[d + 1] ||
               size(coeffs[d], 3) != nbasis[d]
                error("Rank and basis specification does not match " *
                      "coefficient dimensionality")
            end
        end

        return new(ndim, rank, nbasis, coeffs)
    end
end

function show(io::IO, C::FTCoeff)
    @printf(io, "FTCoeff:\n")

    @printf(io, "  ndim: %d\n", C.ndim)

    @printf(io, "  rank: [")
    for d in 1:C.ndim
        @printf(io, "%d, ", C.rank[d])
    end
    @printf(io, "%d]\n", C.rank[C.ndim + 1])

    @printf(io, "  nbasis: [")
    for d in 1:C.ndim - 1
        @printf(io, "%d, ", C.nbasis[d])
    end
    @printf(io, "%d]\n", C.nbasis[C.ndim])

    for d in 1:C.ndim
        rkm1, rk, nk = size(C.coeffs[d])
        @printf(io, "  core %d: [%d×%d×%d] Array{Float64,3}:\n", d, rkm1, rk,
                nk)
        for k = 1:nk
            @printf(io, "    [:, :, %d] =\n", k)
            for i = 1:rkm1
                @printf("     ")
                for j = 1:rk
                    @printf(" % .6e", C.coeffs[d][i, j, k])
                end
                @printf("\n")
            end

            if d < C.ndim || k < nk
                @printf("\n")
            end
        end
    end
end

function (+)(C1::FTCoeff, C2::FTCoeff)
    if C1.ndim != C2.ndim
        error("Dimensions of two FTCoeffs do not match")
    end

    if any(C1.rank .!= C2.rank)
        error("Ranks of two FTCoeffs do not match")
    end

    if any(C1.nbasis .!= C2.nbasis)
        error("Basis expansion of two FTCoeffs do not match")
    end

    return FTCoeff(C1.ndim, C1.rank, C1.nbasis,
                   [C1.coeffs[i] + C2.coeffs[i] for i in 1:C1.ndim])
end

function (-)(C::FTCoeff)
    return FTCoeff(C.ndim, C.rank, C.nbasis, [-C.coeffs[d] for d in 1:C.ndim])
end

function (-)(C1::FTCoeff, C2::FTCoeff)
    return C1 + (-C2)
end

function (*)(α::Real, C::FTCoeff)
    return FTCoeff(C.ndim, C.rank, C.nbasis,
                   [α * C.coeffs[d] for d in 1:C.ndim])
end

function zero(::Type{FTCoeff}, ndim::Int, rank::Array{Int,1},
              nbasis::Array{Int,1})
    coeffs = Array{Array{Float64,3},1}(undef, ndim)
    for d in 1:ndim
        coeffs[d] = zeros(rank[d], rank[d + 1], nbasis[d])
    end

    return FTCoeff(ndim, rank, nbasis, coeffs)
end

function zero(::Type{FTCoeff}, ndim::Int, rank::Int, nbasis::Int)
    rank_ = ones(Int, ndim + 1)
    rank_[2:ndim] .= rank

    nbasis_ = nbasis * ones(Int, ndim)

    return zero(FTCoeff, ndim, rank_, nbasis_)
end

function zero(C::FTCoeff)
    return zero(FTCoeff, C.ndim, C.rank, C.nbasis)
end

function rand(dist::Distribution{Univariate}, ::Type{FTCoeff}, ndim::Int,
              rank::Array{Int,1}, nbasis::Array{Int,1})
    coeffs = Array{Array{Float64,3},1}(undef, ndim)
    for d in 1:ndim
        coeffs[d] = rand(dist, rank[d], rank[d + 1], nbasis[d])
    end

    return FTCoeff(ndim, rank, nbasis, coeffs)
end

function rand(dist::Distribution{Univariate}, ::Type{FTCoeff}, ndim::Int,
              rank::Int, nbasis::Int)
    rank_ = ones(Int, ndim + 1)
    rank_[2:ndim] .= rank

    nbasis_ = nbasis * ones(Int, ndim)

    return rand(dist, FTCoeff, ndim, rank_, nbasis_)
end

function rand(dist::Distribution{Univariate}, C::FTCoeff)
    return rand(dist, FTCoeff, C.ndim, C.rank, C.nbasis)
end

function sign(C::FTCoeff)
    S = zero(C)
    for d = 1:C.ndim
        for n = 1:C.nbasis[d]
            for j = 1:C.rank[d + 1]
                for i = 1:C.rank[d]
                    S.coeffs[d][i, j, n] = sign(C.coeffs[d][i, j, n])
                end
            end
        end
    end

    return S
end

function sumfuncoeffs(ndim::Int)
    rank = 2 * ones(Int, ndim + 1)
    rank[1] = 1
    rank[ndim + 1] = 1

    nbasis = 2 * ones(Int, ndim)

    C = Array{Array{Float64,3},1}(undef, ndim)

    C[1] = zeros(1, 2, 2)
    C[1][1, 1, 2] = 1.0
    C[1][1, 2, 1] = 1.0

    for d in 2:ndim  - 1
        C[d] = zeros(2, 2, 2)
        C[d][1, 1, 1] = 1.0
        C[d][2, 2, 1] = 1.0
        C[d][2, 1, 2] = 1.0
    end

    C[ndim] = zeros(2, 1, 2)
    C[ndim][1, 1, 1] = 1.0
    C[ndim][2, 1, 2] = 1.0

    return FTCoeff(ndim, rank, nbasis, C)
end

function evaluate(x::AbstractArray{Float64,1}, C::FTCoeff)
    if C.ndim != size(x, 1)
        error("Dimension of input and FT coefficients don't match")
    end

    if any(x .<= -1.0) || any(x .>= 1.0)
        error("Input components must be in the interval [-1, 1]")
    end

    u = zeros(C.rank[2])
    for n = 1:C.nbasis[1]
        for j = 1:C.rank[2]
            u[j] += C.coeffs[1][1, j, n] * legendre(x[1], n - 1)
        end
    end

    v = zeros(C.rank[C.ndim])
    for n = 1:C.nbasis[C.ndim]
        for i = 1:C.rank[C.ndim]
            v[i] += C.coeffs[C.ndim][i, 1, n] * legendre(x[C.ndim], n - 1)
        end
    end

    for d = C.ndim  - 1:-1:2
        A = zeros(C.rank[d], C.rank[d + 1])
        for n = 1:C.nbasis[d]
            for j = 1:C.rank[d + 1]
                for i = 1:C.rank[d]
                    A[i, j] += C.coeffs[d][i, j, n] * legendre(x[d], n - 1)
                end
            end
        end

        v = A * v
    end

    return dot(u, v)
end

function gradient(x::AbstractArray{Float64,1}, C::FTCoeff)
    if C.ndim != size(x, 1)
        error("Dimension of input and FT coefficients don't match")
    end

    if any(x .<= -1.0) || any(x .>= 1.0)
        error("Input components must be in the interval [-1, 1]")
    end

    # left to right evaluations
    l2r = Array{Array{Float64,1},1}(undef, C.ndim - 1)

    l2r[1] = zeros(C.rank[2])
    for n = 1:C.nbasis[1]
        for j = 1:C.rank[2]
            l2r[1][j] += C.coeffs[1][1, j, n] * legendre(x[1], n - 1)
        end
    end

    for d = 2:C.ndim - 1
        A = zeros(C.rank[d], C.rank[d + 1])
        for n = 1:C.nbasis[d]
            for j = 1:C.rank[d + 1]
                for i = 1:C.rank[d]
                    A[i, j] += C.coeffs[d][i, j, n] * legendre(x[d], n - 1)
                end
            end
        end

        l2r[d] = A' * l2r[d - 1]
    end

    # right to left evaluations
    r2l = Array{Array{Float64,1},1}(undef, C.ndim - 1)

    r2l[C.ndim - 1] = zeros(C.rank[C.ndim])
    for n = 1:C.nbasis[C.ndim]
        for i = 1:C.rank[C.ndim]
            r2l[C.ndim - 1][i] += C.coeffs[C.ndim][i, 1, n] *
                                  legendre(x[C.ndim], n - 1)
        end
    end

    for d = C.ndim - 1:-1:2
        A = zeros(C.rank[d], C.rank[d + 1])
        for n = 1:C.nbasis[d]
            for j = 1:C.rank[d + 1]
                for i = 1:C.rank[d]
                    A[i, j] += C.coeffs[d][i, j, n] * legendre(x[d], n - 1)
                end
            end
        end

        r2l[d - 1] = A * r2l[d]
    end

    # gradient computation
    G = zero(C)

    for n = 1:C.nbasis[1]
        for j = 1:C.rank[2]
            G.coeffs[1][1, j, n] = legendre(x[1], n - 1) * r2l[1][j]
        end
    end

    for d = 2:C.ndim - 1
        for n = 1:C.nbasis[d]
            for j = 1:C.rank[d + 1]
                for i = 1:C.rank[d]
                    G.coeffs[d][i, j, n] = l2r[d - 1][i] *
                                           legendre(x[d], n - 1) * r2l[d][j]
                end
            end
        end
    end

    for n = 1:C.nbasis[C.ndim]
        for i = 1:C.rank[C.ndim]
            G.coeffs[C.ndim][i, 1, n] = l2r[C.ndim - 1][i] *
                                        legendre(x[C.ndim], n - 1)
        end
    end

    return G
end

end # module
