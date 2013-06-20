#pragma once

#include "helper_math.h"

namespace CRTUtil
{
	float4 cross(float4 a, float4 b);
	inline __device__ float4 reflect(float4 i, float4 n)
	{
		return i - 2.0f * n * (n.x * i.x + n.y * i.y + n.z * i.z + n.w * i.w); // dot(n, i);
	}
}