using Test
include("../src/WJFKNN.jl")
#import WJFKNN:WjfKNN, knnSearch
using WJFParallelTask

data = Vector{Vector{Float64}}()
numClusters = 20
numDims = 10
k = UInt32(10)
clusterCenters = Vector{Vector{Float64}}()
clusterSizes = Vector{Vector{Float64}}()
for i = 1:numClusters
    push!(clusterCenters, rand(Float64, numDims))
    push!(clusterSizes, rand(Float64, numDims))
end

numData = 10000
for i = 1:numData
    clusterID = rand(1:numClusters)
    position = clusterCenters[clusterID] .+
        rand(Float64, numDims) .* clusterSizes[clusterID]
    push!(data, position)
end

function euclidDistSqr(u::Vector{Float64}, v::Vector{Float64})
    return sum((u .- v).^2)
end

function cosDistSqr(data::Vector{Vector{Float64}},
    i::UInt32, j::UInt32)
    return sum(data[i].*data[j])^2 /
        (sum(data[i].*data[i]) * sum(data[j].*data[j]))
end

function baselineKNN(distSqrFun::Function)::Vector{Vector{Tuple{Float64,UInt32}}}
    result = Vector{Vector{Tuple{Float64,UInt32}}}(undef, numData)
    function mapFun(u1::UInt32, u2::UInt32)
        distSqrs = Vector{Tuple{Float64,UInt32}}(undef, numData)
        for i::UInt32 = u1:u2
            for j::UInt32 = 1:numData
                distSqrs[j] = (distSqrFun(data[i], data[j]), j)
            end
            distSqrs[i] = (Inf64, i)
            sort!(distSqrs)
            result[i] = distSqrs[1:k]
        end # i
    end
    mapOnly(UInt32(1),UInt32(numData),UInt32(100),mapFun)
    return result
end

function wjfKNN(distSqrFun::Function)::Vector{Vector{Tuple{Float64,UInt32}}}
    wjfknn = WJFKNN.WjfKNN(data, distSqrFun)
    return WJFKNN.knnSearch(wjfknn, k, 0.5)
end

function computeAccuracy(a::Vector{Vector{Tuple{Float64,UInt32}}}, b::Vector{Vector{Tuple{Float64,UInt32}}})
    accuracy = 0.0
    for i = 1:numData
        sa = Set([x[2] for x in a[i]])
        sb = Set([x[2] for x in b[i]])
        accuracy += length(intersect(sa, sb))
    end
    accuracy /= (numData * k)
    return accuracy
end

#@testset "eclidean distance knn" begin
@time a = baselineKNN(euclidDistSqr)
@time b = wjfKNN(euclidDistSqr)
println("accuracy = $(computeAccuracy(a, b))")
#end
