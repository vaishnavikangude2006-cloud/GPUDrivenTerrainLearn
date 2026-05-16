// ===============================
// GPU Driven Terrain - English Version
// Main Core Compute Shader
// ===============================

#pragma kernel TraverseQuadTree
#pragma kernel BuildPatches
#pragma kernel BuildLodMap

// ========================================
// STRUCTURES
// ========================================

struct NodeDescriptor
{
    uint branch;
};

struct RenderPatch
{
    float2 position;
    uint lod;
    uint4 lodTrans;
};

// ========================================
// BUFFERS
// ========================================

uniform uint PassLOD;

ConsumeStructuredBuffer<uint2> ConsumeNodeList;
AppendStructuredBuffer<uint2> AppendNodeList;
AppendStructuredBuffer<uint3> AppendFinalNodeList;

RWStructuredBuffer<NodeDescriptor> NodeDescriptors;

StructuredBuffer<uint3> FinalNodeList;

AppendStructuredBuffer<RenderPatch> CulledPatchList;

RWTexture2D<float> _LodMap;

// ========================================
// CONSTANTS
// ========================================

#define MAX_TERRAIN_LOD 5

uint nodeIdOffsetLOD[6] =
{
    8525,
    2125,
    525,
    125,
    25,
    0
};

uint nodeCountLOD[6] =
{
    160,
    80,
    40,
    20,
    10,
    5
};

// ========================================
// FUNCTIONS
// ========================================

uint GetNodeId(uint2 nodeLoc, uint lod)
{
    return nodeIdOffsetLOD[lod] +
           nodeLoc.y * nodeCountLOD[lod] +
           nodeLoc.x;
}

// ----------------------------------------

bool EvaluateNode(uint2 nodeLoc, uint lod)
{
    float nodeSize = pow(2, lod) * 64.0;

    float2 nodeCenter =
    (
        nodeLoc * nodeSize
    ) + nodeSize * 0.5;

    float2 cameraXZ = float2(0, 0);

    float distanceToCamera =
        distance(nodeCenter, cameraXZ);

    float factor =
        distanceToCamera / (nodeSize * 2.0);

    return factor < 1.0;
}

// ========================================
// QUAD TREE BUILD
// ========================================

[numthreads(1,1,1)]
void TraverseQuadTree(uint3 id : SV_DispatchThreadID)
{
    uint2 nodeLoc =
        ConsumeNodeList.Consume();

    if (PassLOD > 0 &&
        EvaluateNode(nodeLoc, PassLOD))
    {
        AppendNodeList.Append(nodeLoc * 2);

        AppendNodeList.Append
        (
            nodeLoc * 2 + uint2(1,0)
        );

        AppendNodeList.Append
        (
            nodeLoc * 2 + uint2(0,1)
        );

        AppendNodeList.Append
        (
            nodeLoc * 2 + uint2(1,1)
        );

        uint nodeId =
            GetNodeId(nodeLoc, PassLOD);

        NodeDescriptor desc;
        desc.branch = 1;

        NodeDescriptors[nodeId] = desc;
    }
    else
    {
        AppendFinalNodeList.Append
        (
            uint3(nodeLoc, PassLOD)
        );
    }
}

// ========================================
// CREATE PATCH
// ========================================

RenderPatch CreatePatch
(
    uint3 nodeLoc,
    uint2 patchOffset
)
{
    RenderPatch patch;

    float scale =
        pow(2, nodeLoc.z);

    float patchSize =
        8.0 * scale;

    patch.position =
        (nodeLoc.xy * 64.0 * scale) +
        (patchOffset * patchSize);

    patch.lod = nodeLoc.z;

    patch.lodTrans = uint4(0,0,0,0);

    return patch;
}

// ========================================
// BUILD PATCHES
// ========================================

[numthreads(8,8,1)]
void BuildPatches
(
    uint3 id : SV_DispatchThreadID,
    uint3 groupId : SV_GroupID,
    uint3 groupThreadId : SV_GroupThreadID
)
{
    uint3 nodeLoc =
        FinalNodeList[groupId.x];

    uint2 patchOffset =
        groupThreadId.xy;

    RenderPatch patch =
        CreatePatch(nodeLoc, patchOffset);

    CulledPatchList.Append(patch);
}

// ========================================
// GET SECTOR COUNT
// ========================================

uint GetSectorCountPerNode(uint lod)
{
    return pow(2, lod);
}

// ========================================
// BUILD LOD MAP
// ========================================

[numthreads(8,8,1)]
void BuildLodMap
(
    uint3 id : SV_DispatchThreadID
)
{
    uint2 sectorLoc = id.xy;

    [unroll]
    for(uint lod = MAX_TERRAIN_LOD;
        lod >= 0;
        lod--)
    {
        uint sectorCount =
            GetSectorCountPerNode(lod);

        uint2 nodeLoc =
            sectorLoc / sectorCount;

        uint nodeId =
            GetNodeId(nodeLoc, lod);

        NodeDescriptor desc =
            NodeDescriptors[nodeId];

        if(desc.branch == 0)
        {
            _LodMap[sectorLoc] =
                lod / 5.0;

            return;
        }
    }

    _LodMap[sectorLoc] = 0;
}
