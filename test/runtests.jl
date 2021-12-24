using Test
include("../src/WJFKNN.jl")
#import WJFKNN:WjfKNN, knnSearch
using WJFParallelTask

begin
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

numData = 100000
for i = 1:numData
    clusterID = rand(1:numClusters)
    position = clusterCenters[clusterID] .+
        rand(Float64, numDims) .* clusterSizes[clusterID]
    push!(data, position)
end

normalizedData = [v./sqrt(sum(v.*v)) for v in data]
function euclidDist(u::Vector{Float64}, v::Vector{Float64})
    return sqrt(sum((u .- v).^2))
end

function cosDist(u::Vector{Float64}, v::Vector{Float64})
    return 0.5 - 0.5 * sum(u.*v)
end

function baselineKNN(distFun::Function)::Vector{Vector{Tuple{Float64,UInt32}}}
    result = Vector{Vector{Tuple{Float64,UInt32}}}(undef, numData)
    myData = if distFun == cosDist
        normalizedData
    else
        data
    end
    function mapFun(u1::UInt32, u2::UInt32)
        dists = Vector{Tuple{Float64,UInt32}}(undef, numData)
        for i::UInt32 = u1:u2
            for j::UInt32 = 1:numData
                dists[j] = (distFun(myData[i], myData[j]), j)
            end
            dists[i] = (Inf64, i)
            sort!(dists)
            result[i] = dists[1:k]
        end # i
    end
    mapOnly(UInt32(1),UInt32(numData),UInt32(100),mapFun)
    return result
end

function wjfKNNEuclid()::Vector{Vector{Tuple{Float64,UInt32}}}
    wjfKnn = WJFKNN.WjfKNN(data, euclidDist)
    return WJFKNN.knnSearch(wjfKnn, k, 0.5)
end

function wjfKNNCos()::Vector{Vector{Tuple{Float64,UInt32}}}
    wjfKnn = WJFKNN.WjfKNN(normalizedData, cosDist)
    return WJFKNN.knnSearch(wjfKnn, k, 0.5)
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

@testset "cos distance knn" begin
    @time a = baselineKNN(cosDist)
    for round = 1:10
        println("round = $round")
        @time b = wjfKNNCos()
        @test computeAccuracy(a, b) > 0.85
    end
end

@testset "euclid distance knn" begin
    @time a = baselineKNN(euclidDist)
    for round = 1:10
        println("round = $round")
        @time b = wjfKNNEuclid()
        @test computeAccuracy(a, b) > 0.9
    end
end

end
