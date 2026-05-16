// ======================================================
// GPU DRIVEN TERRAIN - ENGLISH VERSION
// ======================================================

// ------------------------------------------------------
// QUADTREE BUFFERS
// ------------------------------------------------------

uniform uint PassLOD;

ConsumeStructuredBuffer<uint2> ConsumeNodeList;
AppendStructuredBuffer<uint2> AppendNodeList;
AppendStructuredBuffer<uint3> AppendFinalNodeList;


// ------------------------------------------------------
// QUADTREE TRAVERSAL
// ------------------------------------------------------

[numthreads(1,1,1)]
void TraverseQuadTree(uint3 id : SV_DispatchThreadID)
{
    uint2 nodeLoc = ConsumeNodeList.Consume();

    if (PassLOD > 0 && EvaluateNode(nodeLoc, PassLOD))
    {
        // Divide current node into 4 child nodes

        AppendNodeList.Append(nodeLoc * 2);
        AppendNodeList.Append(nodeLoc * 2 + uint2(1,0));
        AppendNodeList.Append(nodeLoc * 2 + uint2(0,1));
        AppendNodeList.Append(nodeLoc * 2 + uint2(1,1));
    }
    else
    {
        // Final node for rendering
        AppendFinalNodeList.Append(uint3(nodeLoc, PassLOD));
    }
}


// ------------------------------------------------------
// LOD EVALUATION
// ------------------------------------------------------

// f = d / (n * c)
//
// d = distance from camera
// n = node size
// c = control factor
//
// if f < 1 → split node


// ------------------------------------------------------
// NODE DESCRIPTOR
// ------------------------------------------------------

struct NodeDescriptor
{
    uint branch;
};


// ------------------------------------------------------
// NODE ID CALCULATION
// ------------------------------------------------------

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

uint3 nodeLoc;

uint nodeId =
nodeIdOffsetLOD[nodeLoc.z]
+ nodeLoc.y * nodeCount
+ nodeLoc.x;


// ------------------------------------------------------
// NODE DESCRIPTOR BUFFER
// ------------------------------------------------------

RWStructuredBuffer<NodeDescriptor> NodeDescriptors;


// ------------------------------------------------------
// RENDER PATCH STRUCTURE
// ------------------------------------------------------

struct RenderPatch
{
    float2 position; // world position
    uint lod;        // lod level
};


// ------------------------------------------------------
// BUILD PATCHES
// ------------------------------------------------------

[numthreads(8,8,1)]
void BuildPatches(
    uint3 id : SV_DispatchThreadID,
    uint3 groupId : SV_GroupID,
    uint3 groupThreadId : SV_GroupThreadID)
{
    uint3 nodeLoc = FinalNodeList[groupId.x];

    uint2 patchOffset = groupThreadId.xy;

    // Create terrain patch
    RenderPatch patch =
    CreatePatch(nodeLoc, patchOffset);

    CulledPatchList.Append(patch);
}


// ------------------------------------------------------
// TERRAIN SHADER
// ------------------------------------------------------

StructuredBuffer<RenderPatch> PatchList;

struct appdata
{
    float4 vertex : POSITION;
    float2 uv : TEXCOORD0;

    uint instanceID : SV_InstanceID;
};


// ------------------------------------------------------
// PATCH POSITION + LOD SCALE
// ------------------------------------------------------

RenderPatch patch = PatchList[v.instanceID];

uint lod = patch.lod;

float scale = pow(2, lod);

inVertex.xz *= scale;

inVertex.xz += patch.position;

o.vertex = TransformObjectToHClip(inVertex.xyz);


// ------------------------------------------------------
// HEIGHT MAP UV
// ------------------------------------------------------

float2 heightUV =
(
    inVertex.xz
    + (_WorldSize.xz * 0.5)
    + 0.5
)
/
(_WorldSize.xz + 1);


// ------------------------------------------------------
// APPLY HEIGHTMAP
// ------------------------------------------------------

float height =
tex2Dlod(
    _HeightMap,
    float4(heightUV,0,0)
).r;

inVertex.y =
height * _WorldSize.y;


// ------------------------------------------------------
// FRUSTUM CULLING
// ------------------------------------------------------

bool IsOutSidePlane(
    float4 plane,
    float3 position)
{
    return dot(plane.xyz, position)
    + plane.w < 0;
}


// ------------------------------------------------------
// FULL FRUSTUM CHECK
// ------------------------------------------------------

bool FrustumCull(
    float4 planes[6],
    Bounds bounds)
{
    return
    IsBoundsOutSidePlane(planes[0],bounds) ||
    IsBoundsOutSidePlane(planes[1],bounds) ||
    IsBoundsOutSidePlane(planes[2],bounds) ||
    IsBoundsOutSidePlane(planes[3],bounds) ||
    IsBoundsOutSidePlane(planes[4],bounds) ||
    IsBoundsOutSidePlane(planes[5],bounds);
}


// ------------------------------------------------------
// HIZ OCCLUSION CULLING
// ------------------------------------------------------

bool HizOcclusionCull(Bounds bounds)
{
    Bounds boundsUVD =
    GetBoundsUVD(bounds);

    float3 minP =
    boundsUVD.minPosition;

    float3 maxP =
    boundsUVD.maxPosition;

    uint mip =
    GetHizMip(boundsUVD);

    float d1 =
    _HizMap.SampleLevel(
        _point_clamp_sampler,
        minP.xy,
        mip
    ).r;

    float d2 =
    _HizMap.SampleLevel(
        _point_clamp_sampler,
        maxP.xy,
        mip
    ).r;

    #if _REVERSE_Z

    float depth = maxP.z;

    return d1 > depth
        && d2 > depth;

    #else

    float depth = minP.z;

    return d1 < depth
        && d2 < depth;

    #endif
}


// ------------------------------------------------------
// BUILD LOD MAP
// ------------------------------------------------------

[numthreads(8,8,1)]
void BuildLodMap(
    uint3 id : SV_DispatchThreadID)
{
    uint2 sectorLoc = id.xy;

    for(
        uint lod = MAX_TERRAIN_LOD;
        lod >= 0;
        lod --
    )
    {
        uint sectorCount =
        GetSectorCountPerNode(lod);

        uint2 nodeLoc =
        sectorLoc / sectorCount;

        uint nodeId =
        GetNodeId(nodeLoc,lod);

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


// ------------------------------------------------------
// LOD TRANSITIONS
// ------------------------------------------------------

struct RenderPatchLOD
{
    float2 position;

    uint lod;

    uint4 lodTrans;
};


// ------------------------------------------------------
// SEAM FIX FUNCTION
// ------------------------------------------------------

// Fixes gaps between
// different LOD terrain patches

void FixLODConnectSeam()
{
    // Adjust border vertices
    // to remove terrain cracks
}


// ======================================================
// END OF GPU DRIVEN TERRAIN CODE
// ======================================================
