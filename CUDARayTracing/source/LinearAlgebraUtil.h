#pragma once

#include "helper_math.h"

namespace CRTUtil
{
	float4 cross(float4 a, float4 b);
	inline __device__ __host__ float4 reflect(float4 i, float4 n)
	{
		return i - 2.0f * n * dot(n, i);
	}
}