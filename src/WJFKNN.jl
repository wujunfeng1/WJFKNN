module WJFKNN
export WjfKNN, knnSearch
using WJFParallelTask

function findCenter(
    data::Vector,
    items::Vector{UInt32},
    maxNumSamples::UInt32,
    distSqrFun::Function,
)::UInt32
    numItems = length(items)
    numSamples = min(UInt32(ceil(sqrt(numItems))), maxNumSamples)
    samples::Set{UInt32} = Set{UInt32}()
    for idxSample = 1:numSamples
        sample = (rand(UInt32) % numItems) + 1
        while sample in samples
            sample = (rand(UInt32) % numItems) + 1
        end
        push!(samples, sample)
    end
    sampleLists::Vector{UInt32} = [sample for sample in samples]

    function mapFun(u0::UInt32, u1::UInt32)::Tuple{Float64, UInt32}
        mySmallestSumDSqr = Inf64
        myBestSample = UInt32(0)
        for u = u0:u1
            sumDSqr::Float64 = 0.0
            for v = 1:numSamples
                sumDSqr += distSqrFun(data[sampleLists[u]], data[sampleLists[v]])
            end
            if sumDSqr < mySmallestSumDSqr
                mySmallestSumDSqr = sumDSqr
                myBestSample = sampleLists[u]
            end
        end # u
        return (mySmallestSumDSqr, myBestSample)
    end

    function reduceFun(values::Vector{Tuple{Float64, UInt32}}
        )::Tuple{Float64, UInt32}
        mySmallestSumDSqr = Inf64
        myBestSample = UInt32(0)
        for (sumDSqr, sample) in values
            if sumDSqr < mySmallestSumDSqr
                mySmallestSumDSqr = sumDSqr
                myBestSample = sample
            end
        end
        return (mySmallestSumDSqr, myBestSample)
    end

    (smallestSumDSqr, bestSample) = mapReduce(
        UInt32(1), numSamples, UInt32(100),
        mapFun, reduceFun, (Inf64, UInt32(0)))

    @assert bestSample > 0
    return items[bestSample]
end

function findFarthestItem(
    data::Vector,
    items::Vector{UInt32},
    center,
    subCenters::Vector{UInt32},
    distSqrFun::Function,
)::UInt32
    function mapFun(u0::UInt32, u1::UInt32)::Tuple{Float64,UInt32}
        myFarthestDistSqr = 0.0
        myFarthestItem = 0
        for u = u0:u1
            item = items[u]
            myDistSqr = distSqrFun(data[item], center)
            for j in subCenters
                myDistSqr = min(myDistSqr, distSqrFun(data[item], data[j]))
            end
            if myDistSqr > myFarthestDistSqr
                myFarthestItem = items[u]
                myFarthestDistSqr = myDistSqr
            end
        end # u
        return (myFarthestDistSqr, myFarthestItem)
    end

    function reduceFun(values::Vector{Tuple{Float64,UInt32}}
        )::Tuple{Float64,UInt32}
        myFarthestDistSqr = 0.0
        myFarthestItem = 0
        for (distSqr, item) in values
            if distSqr > myFarthestDistSqr
                myFarthestDistSqr = distSqr
                myFarthestItem = item
            end
        end
        return (myFarthestDistSqr, myFarthestItem)
    end

    numItems = length(items)
    (farthestDistSqr, farthestItem) = mapReduce(
        UInt32(1), UInt32(numItems), UInt32(100),
        mapFun, reduceFun, (0.0, UInt32(0))
    )
    @assert farthestItem != 0
    return farthestItem
end

function findSubCenters(
    data::Vector,
    items::Vector{UInt32},
    maxNumSamples::UInt32,
    numPartitionsPerNode::UInt32,
    distSqrFun::Function
)::Vector{UInt32}
    center = data[findCenter(data, items, maxNumSamples, distSqrFun)]
    subCenters = Vector{UInt32}()
    while length(subCenters) < numPartitionsPerNode
        push!(subCenters, findFarthestItem(data, items, center, subCenters, distSqrFun))
    end
    return subCenters
end

function computeSubCoverSetInfos(
    data::Vector,
    items::Vector{UInt32},
    subCenters::Vector{UInt32},
    distSqrFun::Function,
)::Vector{Tuple{UInt32,UInt32,Float64,Float64}}
    numItems = length(items)
    subCoverSetInfos::Vector{Tuple{Int32,Int32,Float64,Float64}} =
        Vector{Tuple{Int32,Int32,Float64,Float64}}(undef, numItems)
    function mapFun(u0::UInt32, u1::UInt32)
        mySubCenterDists = fill(0.0, (length(subCenters),))
        mySubCenterRanks = Vector{UInt32}(undef, length(subCenters))
        for u = u0:u1
            item = items[u]
            for i::UInt32 = 1:length(subCenters)
                mySubCenterRanks[i] = i
                mySubCenterDists[i] =
                    distSqrFun(data[item], data[subCenters[i]])
            end
            sort!(mySubCenterRanks, by = x -> mySubCenterDists[x])
            subCoverSetInfos[u] = (
                mySubCenterRanks[1],
                mySubCenterRanks[2],
                mySubCenterDists[mySubCenterRanks[1]],
                mySubCenterDists[mySubCenterRanks[2]],
            )
        end # u
    end
    mapOnly(UInt32(1), UInt32(numItems), UInt32(100), mapFun)
    return subCoverSetInfos
end

function getSubCoverItems(
    items::Vector{UInt32},
    subCoverSetInfos::Vector{Tuple{UInt32,UInt32,Float64,Float64}},
    numPartitionsPerNode::UInt32,
)::Vector{Vector{UInt32}}
    result::Vector{Vector{UInt32}} = Vector{Vector{UInt32}}()
    for i = 1:numPartitionsPerNode
        push!(result, Vector{UInt32}())
    end
    for i = 1:length(items)
        (j1, j2, d1, d2) = subCoverSetInfos[i]
        push!(result[j1], items[i])
        push!(result[j2], items[i])
    end
    return result
end

struct CoverSet
    mapping::Dict{UInt32,UInt32}
    subCenters::Vector{UInt32}
    subCoverSetInfos::Vector{Tuple{UInt32,UInt32,Float64,Float64}}
    subCoverSets::Vector{CoverSet}
end

function CoverSet(
    data::Vector,
    items::Vector{UInt32},
    maxNumSamples::UInt32,
    numPartitionsPerNode::UInt32,
    distSqrFun::Function
)
    mapping::Dict{UInt32,UInt32} = Dict{UInt32,UInt32}()
    for i = 1:length(items)
        mapping[items[i]] = i
    end
    if length(items) <= maxNumSamples
        return CoverSet(
        mapping,
        Vector{UInt32}(),
        Vector{Tuple{UInt32,UInt32,Float64,Float64}}(),
        Vector{CoverSet}())
    else
        subCenters = findSubCenters(
            data,
            items,
            maxNumSamples,
            numPartitionsPerNode,
            distSqrFun
        )
        subCoverSetInfos =
            computeSubCoverSetInfos(data, items, subCenters, distSqrFun)
        subCoverItems::Vector{Vector{UInt32}} =
            getSubCoverItems(items, subCoverSetInfos, numPartitionsPerNode)
        subCoverSets = [
            CoverSet(
                data,
                subCoverItems[i],
                maxNumSamples,
                numPartitionsPerNode,
                distSqrFun
            ) for i = 1:numPartitionsPerNode
        ]
        return CoverSet(mapping, subCenters, subCoverSetInfos, subCoverSets)
    end
end

struct WjfKNN
    data::Vector
    distSqrFun::Function
    maxNumSamples::UInt32
    numPartitionsPerNode::UInt32
    entry::CoverSet
end

function WjfKNN(
    data::Vector,
    distSqrFun::Function,
    maxNumSamples::UInt32 = UInt32(100),
    numPartitionsPerNode::UInt32 = UInt32(16),
)
    return WjfKNN(
        data,
        distSqrFun,
        maxNumSamples,
        numPartitionsPerNode,
        CoverSet(
            data,
            [i for i::UInt32 = 1:length(data)],
            maxNumSamples,
            numPartitionsPerNode,
            distSqrFun
        ),
    )
end

function getNeighbors(coverSet::CoverSet, i::UInt32,
    bisearchThres::Float64)::Set{UInt32}
    if length(coverSet.subCoverSets) != 0
        (j1, j2, d1, d2) = coverSet.subCoverSetInfos[coverSet.mapping[i]]
        res::Set{UInt32} = getNeighbors(coverSet.subCoverSets[j1],
            i, bisearchThres)
        if d2 * bisearchThres <= d1
            res = union(res,
                getNeighbors(coverSet.subCoverSets[j2], i, bisearchThres))
        end
        return res
    else
        res = Set{UInt32}(keys(coverSet.mapping))
        return res
    end
end

function knnSearch(wjfknn::WjfKNN, i::UInt32, k::UInt32,
    bisearchThres::Float64)::Vector{Tuple{Float64, UInt32}}
    neighbors::Set{UInt32} = getNeighbors(wjfknn.entry, i, bisearchThres)
    results::Vector{Tuple{Float64, UInt32}} = Vector{Tuple{Float64, UInt32}}()
    for neighbor in neighbors
        if neighbor == i
            continue
        end
        distSqr = wjfknn.distSqrFun(wjfknn.data[i], wjfknn.data[neighbor])
        result = (distSqr, neighbor)
        insert!(results, searchsortedfirst(results, result), result)
        if length(results) > k
            pop!(results)
        end
    end
    return results
end

function knnSearch(
    wjfknn::WjfKNN, k::UInt32, bisearchThres::Float64
    )::Vector{Vector{Tuple{Float64, UInt32}}}
    numData = length(wjfknn.data)
    result::Vector{Vector{Tuple{Float64, UInt32}}} =
        Vector{Vector{Tuple{Float64, UInt32}}}(undef, numData)
    function mapFun(u1::UInt32, u2::UInt32)
        for u::UInt32 = u1:u2
            result[u] = knnSearch(wjfknn, u, k, bisearchThres)
        end # u
    end
    mapOnly(UInt32(1), UInt32(numData), UInt32(100), mapFun)
    return result
end

end # module
