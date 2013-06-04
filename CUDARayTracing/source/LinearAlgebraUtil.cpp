#include "LinearAlgebraUtil.h"

namespace CRTUtil
{
	float4 cross(float4 a, float4 b)
	{
		const float3 temp = ::cross(make_float3(a.x, a.y, a.z), make_float3(b.x, b.y, b.z));
		return make_float4(temp.x, temp.y, temp.z, 0);
	}
}