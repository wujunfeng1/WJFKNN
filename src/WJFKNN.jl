module WJFKNN
export WjfKNN, knnSearch
using WJFParallelTask

struct CoverSet
    mapping::Dict{UInt32,UInt32}
    subCenters::Vector{UInt32}
    subCoverSetInfos::Vector{Tuple{UInt32,UInt32,Float64,Float64}}
    subCoverSets::Vector{CoverSet}
end

function findCenterSequentially(
    data::Vector,
    items::Vector{UInt32},
    maxNumSamples::UInt32,
    distFun::Function,
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

    mySmallestSumD = Inf64
    myBestSample = UInt32(0)
    for u::UInt32 = 1:UInt32(numSamples)
        sumD::Float64 = 0.0
        for v = 1:numSamples
            sumD += distFun(data[sampleLists[u]], data[sampleLists[v]])
        end
        if sumD < mySmallestSumD
            mySmallestSumD = sumD
            myBestSample = sampleLists[u]
        end
    end
    return items[myBestSample]
end

function findCenter(
    data::Vector,
    items::Vector{UInt32},
    maxNumSamples::UInt32,
    distFun::Function,
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
        mySmallestSumD = Inf64
        myBestSample = UInt32(0)
        for u = u0:u1
            sumD::Float64 = 0.0
            for v = 1:numSamples
                sumD += distFun(data[sampleLists[u]], data[sampleLists[v]])
            end
            if sumD < mySmallestSumD
                mySmallestSumD = sumD
                myBestSample = sampleLists[u]
            end
        end # u
        return (mySmallestSumD, myBestSample)
    end

    function reduceFun(values::Vector{Tuple{Float64, UInt32}}
        )::Tuple{Float64, UInt32}
        mySmallestSumD = Inf64
        myBestSample = UInt32(0)
        for (sumD, sample) in values
            if sumD < mySmallestSumD
                mySmallestSumD = sumD
                myBestSample = sample
            end
        end
        return (mySmallestSumD, myBestSample)
    end

    bestSample = 0
    while bestSample == 0
        (smallestSumD, bestSample) = mapReduce(
            UInt32(1), numSamples, UInt32(100),
            mapFun, reduceFun, (Inf64, UInt32(0)))
    end

    return items[bestSample]
end

function findFarthestItemSequentially(
    data::Vector,
    items::Vector{UInt32},
    center,
    subCenters::Vector{UInt32},
    distFun::Function,
)::UInt32
    numItems = length(items)
    myFarthestdist = 0.0
    myFarthestItem = 0
    for u::UInt32 = 1:UInt32(numItems)
        item = items[u]
        mydist = distFun(data[item], center)
        for j in subCenters
            mydist = min(mydist, distFun(data[item], data[j]))
        end
        if mydist > myFarthestdist
            myFarthestItem = items[u]
            myFarthestdist = mydist
        end
    end # u
    return myFarthestItem
end

function findFarthestItem(
    data::Vector,
    items::Vector{UInt32},
    center,
    subCenters::Vector{UInt32},
    distFun::Function,
)::UInt32
    function mapFun(u0::UInt32, u1::UInt32)::Tuple{Float64,UInt32}
        myFarthestdist = 0.0
        myFarthestItem = 0
        for u = u0:u1
            item = items[u]
            mydist = distFun(data[item], center)
            for j in subCenters
                mydist = min(mydist, distFun(data[item], data[j]))
            end
            if mydist > myFarthestdist
                myFarthestItem = items[u]
                myFarthestdist = mydist
            end
        end # u
        return (myFarthestdist, myFarthestItem)
    end

    function reduceFun(values::Vector{Tuple{Float64,UInt32}}
        )::Tuple{Float64,UInt32}
        myFarthestdist = 0.0
        myFarthestItem = 0
        for (dist, item) in values
            if dist > myFarthestdist
                myFarthestdist = dist
                myFarthestItem = item
            end
        end
        return (myFarthestdist, myFarthestItem)
    end

    numItems = length(items)
    farthestItem = 0
    while farthestItem == 0
        (farthestdist, farthestItem) = mapReduce(
            UInt32(1), UInt32(numItems), UInt32(100),
            mapFun, reduceFun, (0.0, UInt32(0))
        )
    end
    return farthestItem
end

function fillSubCenters!(
    subCenters::Vector{UInt32},
    data::Vector,
    items::Vector{UInt32},
    maxNumSamples::UInt32,
    numPartitionsPerNode::UInt32,
    distFun::Function
)
    center = data[findCenterSequentially(data, items, maxNumSamples, distFun)]
    while length(subCenters) < numPartitionsPerNode
        push!(subCenters, findFarthestItemSequentially(
            data, items, center, subCenters, distFun))
    end
end

function findSubCenters(
    data::Vector,
    items::Vector{UInt32},
    maxNumSamples::UInt32,
    numPartitionsPerNode::UInt32,
    distFun::Function
)::Vector{UInt32}
    center = data[findCenter(data, items, maxNumSamples, distFun)]
    subCenters = Vector{UInt32}()
    while length(subCenters) < numPartitionsPerNode
        push!(subCenters, findFarthestItem(data, items, center, subCenters, distFun))
    end
    return subCenters
end

function fillSubCoverSetInfos!(
    subCoverSetInfos::Vector{Tuple{UInt32,UInt32,Float64,Float64}},
    data::Vector,
    items::Vector{UInt32},
    subCenters::Vector{UInt32},
    distFun::Function,
)
    numItems = length(items)
    mySubCenterDists = fill(0.0, (length(subCenters),))
    mySubCenterRanks = Vector{UInt32}(undef, length(subCenters))
    for u::UInt32 = 1:UInt32(numItems)
        item = items[u]
        for i::UInt32 = 1:length(subCenters)
            mySubCenterRanks[i] = i
            mySubCenterDists[i] =
                distFun(data[item], data[subCenters[i]])
        end
        sort!(mySubCenterRanks, by = x -> mySubCenterDists[x])
        subCoverSetInfos[u] = (
            mySubCenterRanks[1],
            mySubCenterRanks[2],
            mySubCenterDists[mySubCenterRanks[1]],
            mySubCenterDists[mySubCenterRanks[2]],
        )
    end
end

function computeSubCoverSetInfos(
    data::Vector,
    items::Vector{UInt32},
    subCenters::Vector{UInt32},
    distFun::Function,
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
                    distFun(data[item], data[subCenters[i]])
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

function fillCoverSet!(
    coverSet::CoverSet,
    data::Vector,
    items::Vector{UInt32},
    maxNumSamples::UInt32,
    numPartitionsPerNode::UInt32,
    distFun::Function,
    smallSubCoverSets::Vector{Tuple{CoverSet, Vector{UInt32}}},
)
    fillSubCenters!(
        coverSet.subCenters,
        data,
        items,
        maxNumSamples,
        numPartitionsPerNode,
        distFun
    )
    fillSubCoverSetInfos!(
        coverSet.subCoverSetInfos,
        data,
        items,
        coverSet.subCenters,
        distFun,
    )
    subCoverItems::Vector{Vector{UInt32}} =
        getSubCoverItems(items, coverSet.subCoverSetInfos, numPartitionsPerNode)
    for i = 1:numPartitionsPerNode
        coverSet.subCoverSets[i] = CoverSet(
            data,
            subCoverItems[i],
            maxNumSamples,
            numPartitionsPerNode,
            distFun,
            smallSubCoverSets
        )
    end
end

function CoverSet(
    data::Vector,
    items::Vector{UInt32},
    maxNumSamples::UInt32,
    numPartitionsPerNode::UInt32,
    distFun::Function,
    smallSubCoverSets::Vector{Tuple{CoverSet, Vector{UInt32}}},
)
    numItems = length(items)
    numData = length(data)
    mapping::Dict{UInt32,UInt32} = Dict{UInt32,UInt32}()
    for i = 1:numItems
        mapping[items[i]] = i
    end
    smallSubCoverSetSize = maxNumSamples * UInt32(sqrt(numPartitionsPerNode))
    if length(items) <= maxNumSamples
        return CoverSet(
        mapping,
        Vector{UInt32}(),
        Vector{Tuple{UInt32,UInt32,Float64,Float64}}(),
        Vector{CoverSet}())
    elseif numData > numItems && numItems <= smallSubCoverSetSize
        coverSet = CoverSet(
        mapping,
        Vector{UInt32}(),
        Vector{Tuple{UInt32,UInt32,Float64,Float64}}(undef, numItems),
        Vector{CoverSet}(undef, numPartitionsPerNode))
        push!(smallSubCoverSets, (coverSet,items))
        return coverSet
    else
        subCenters = findSubCenters(
            data,
            items,
            maxNumSamples,
            numPartitionsPerNode,
            distFun
        )
        subCoverSetInfos =
            computeSubCoverSetInfos(data, items, subCenters, distFun)
        subCoverItems::Vector{Vector{UInt32}} =
            getSubCoverItems(items, subCoverSetInfos, numPartitionsPerNode)
        subCoverSets = [
            CoverSet(
                data,
                subCoverItems[i],
                maxNumSamples,
                numPartitionsPerNode,
                distFun,
                smallSubCoverSets
            ) for i = 1:numPartitionsPerNode
        ]
        return CoverSet(mapping, subCenters, subCoverSetInfos, subCoverSets)
    end
end

struct WjfKNN
    data::Vector
    distFun::Function
    maxNumSamples::UInt32
    numPartitionsPerNode::UInt32
    entry::CoverSet
end

function finishSmallSubCoverSets(
    smallSubCoverSets::Vector{Tuple{CoverSet, Vector{UInt32}}},
    data::Vector,
    maxNumSamples::UInt32,
    numPartitionsPerNode::UInt32,
    distFun::Function,
    )::Vector{Tuple{CoverSet, Vector{UInt32}}}
    function mapFun(u0::UInt32,u1::UInt32)
        myResult = Vector{Tuple{CoverSet, Vector{UInt32}}}()
        for u in u0:u1
            subCoverSet = smallSubCoverSets[u][1]
            items = smallSubCoverSets[u][2]
            fillCoverSet!(
                subCoverSet,
                data,
                items,
                maxNumSamples,
                numPartitionsPerNode,
                distFun,
                myResult,
                )
        end
        return myResult
    end
    function blockPrefixFun(x0,xs)
        return xs
    end
    result = mapPrefix(
        UInt32(1),
        UInt32(length(smallSubCoverSets)),
        UInt32(1),
        mapFun,
        blockPrefixFun,
        Vector{Tuple{CoverSet, Vector{UInt32}}}(),
    )
    return result
end

function WjfKNN(
    data::Vector,
    distFun::Function,
    maxNumSamples::UInt32 = UInt32(256),
    numPartitionsPerNode::UInt32 = UInt32(16),
)
    smallSubCoverSets = Vector{Tuple{CoverSet, Vector{UInt32}}}()
    entry = CoverSet(
        data,
        [i for i::UInt32 = 1:length(data)],
        maxNumSamples,
        numPartitionsPerNode,
        distFun,
        smallSubCoverSets,
    )
    while length(smallSubCoverSets) > 0
        smallSubCoverSets = finishSmallSubCoverSets(
            smallSubCoverSets,
            data,
            maxNumSamples,
            numPartitionsPerNode,
            distFun,
        )
    end
    return WjfKNN(
        data,
        distFun,
        maxNumSamples,
        numPartitionsPerNode,
        entry,
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
        dist = wjfknn.distFun(wjfknn.data[i], wjfknn.data[neighbor])
        result = (dist, neighbor)
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
