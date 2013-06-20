////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//#include <stdio.h>
#include "RayTracer.h"
//#include "helper_cuda.h"


__device__ float SphereIntersection(float4 rayOrigin, float4 rayDirection, float4 spherePosition, float sphereRadius);
__device__ float QuadatricSolver(float A, float B, float C);
__device__ float4 PointLightContribution(float4 position, float4 normal, float4 color, float4 lightPosition, float4 cameraPosition);


__global__ void RayTracer(uchar4* dest, const int imageW, const int imageH, float4 cameraPosition, float4 cameraUp, float4 cameraForward, float4 cameraRight, float nearPlaneDistance, float2 viewSize)
{
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y;

	// Compute the location in the dest array that will be written to
	const int pixelIndex = imageW * iy + ix;
	float4 pixelColor;

	// Compute the center of the near plane. All rays will be computed as an offset from this point
	const float4 lookAt = cameraPosition + cameraForward * nearPlaneDistance;

	// Find where the ray intersects the near plane and create the vector portion of the ray from that
	const float4 rayMidPoint = lookAt + cameraRight * ((float(ix) / float(imageW) - 0.5f) * viewSize.x) + cameraUp * ((float(iy) / float(imageH) - 0.5f) * viewSize.y); 
	float4 ray = normalize(rayMidPoint - cameraPosition);

	// Hardcoded sphere
	const float4 sphereCenter = make_float4(0, -1000, 50, 1);
	const float4 sphereColor = make_float4(0.4f, 0, 0.4f, 1.0f);
	const float radius = 1000.0f;

	const float4 otherSphereCenter = make_float4(0, 5, 30, 1);
	const float4 otherSphereColor = make_float4(0, 0.4f, 0.4f, 1.0f);
	const float otherRadius = 1.0f;

	// Hardcoded light
	const float4 lightPosition = make_float4(0, 30, 25, 1);

	// Check if the camera can see the two spheres
	float t = SphereIntersection(cameraPosition, ray, sphereCenter, radius);
	float otherT = SphereIntersection(cameraPosition, ray, otherSphereCenter, otherRadius);

	float4 intersectionPoint; 
	float4 intersectionNormal;

	// If the first sphere is closer
	if(t > 0 && (t < otherT || otherT == -1.0f))
	{
		intersectionPoint = cameraPosition + t * ray;
		intersectionNormal = normalize(intersectionPoint - sphereCenter);
		float4 reflectedRay = CRTUtil::reflect(ray, intersectionNormal);
		
		ray = normalize(lightPosition - intersectionPoint);

		// Check if there is anything between the first sphere and the light
		float lightT = SphereIntersection(intersectionPoint, ray, otherSphereCenter, otherRadius);
		float reflectT = SphereIntersection(intersectionPoint, reflectedRay, otherSphereCenter, otherRadius);
		

		if(lightT <= 0)
		{
			if(reflectT > 0)
			{
				const float4 reflectionIntersectionPoint = intersectionPoint + reflectedRay * reflectT;
				const float4 reflectionIntersectionNormal = reflectionIntersectionPoint - otherSphereCenter;
				float4 litOtherSphereColor = PointLightContribution(reflectionIntersectionPoint, reflectionIntersectionNormal, otherSphereColor, lightPosition, cameraPosition);
				// Didn't compile
				//pixelColor = sphereColor * 0.7f + otherSphereColor * 0.3f;
				pixelColor = sphereColor * 0.7f;
				pixelColor += litOtherSphereColor * 0.3f;
			}
			else
			{
				pixelColor = sphereColor * 1.0f;
				pixelColor += make_float4(BACKGROUND_COLOR) * 0.0f;
			}
			
			// If not, light it fully
			pixelColor = PointLightContribution(intersectionPoint, intersectionNormal, pixelColor, lightPosition, cameraPosition);
		}
		else
		{
			//intersectionPoint = intersectionPoint + lightT * ray;
			//intersectionNormal = normalize(intersectionPoint - otherSphereCenter);

			//pixelColor = PointLightContribution(intersectionPoint, intersectionNormal, otherSphereColor, lightPosition, cameraPosition);

			// Otherwise it is shadowed, just use ambient light
			pixelColor = sphereColor * AMBIENT_STRENGTH;
			pixelColor.w = 1.0f;
		}
	}
	else if(otherT > 0)
	{
		intersectionPoint = cameraPosition + otherT * ray;
		intersectionNormal = normalize(intersectionPoint - otherSphereCenter);

		pixelColor = PointLightContribution(intersectionPoint, intersectionNormal, otherSphereColor, lightPosition, cameraPosition);
	}
	else
	{
		pixelColor = make_float4(BACKGROUND_COLOR);
	}

	dest[pixelIndex] = make_uchar4((unsigned char)(pixelColor.x * 255), (unsigned char)(pixelColor.y * 255), (unsigned char)(pixelColor.z * 255), 255);
}

__device__ float4 PointLightContribution(float4 position, float4 normal, float4 color, float4 lightPosition, float4 cameraPosition)
{
		const float4 lightDirection = normalize(lightPosition - position);
		const float4 halfVector = normalize(lightDirection + normalize(cameraPosition - position));
		float diffuseStrength = dot(normal, lightDirection);
		float specularStrength = dot(normal, halfVector);
		diffuseStrength = clamp(diffuseStrength, 0.0f, 1.0f);
		specularStrength = clamp(specularStrength, 0.0f, 1.0f);
		specularStrength = pow(specularStrength, 15);
		float lightCoefficient = diffuseStrength + AMBIENT_STRENGTH;

		const float4 litColor = make_float4(clamp(color.x * lightCoefficient + specularStrength, 0.0f, 1.0f), 
											clamp(color.y * lightCoefficient + specularStrength, 0.0f, 1.0f),
											clamp(color.z * lightCoefficient + specularStrength, 0.0f, 1.0f),
											1.0f);
		return litColor;
}

__device__ float SphereIntersection(float4 rayOrigin, float4 rayDirection, float4 spherePosition, float sphereRadius)
{
	// Calculate the three coefficients in the quadratic equation
	const float4 rayOriginMinusSphereCenter = rayOrigin - spherePosition;

	const float A = dot(rayDirection, rayDirection);
	const float B = 2 * dot(rayOriginMinusSphereCenter, rayDirection);
	const float C = dot(rayOriginMinusSphereCenter, rayOriginMinusSphereCenter) - sphereRadius * sphereRadius;

	return QuadatricSolver(A, B, C);
}

__device__ float QuadatricSolver(float A, float B, float C)
{
	//Calculate the discriminant
	const float disc = B * B - 4 * A * C;

	float t = -1.0f;

	if(disc >= 0)
	{
		const float discSqrt = sqrtf(disc);
		float q;
		
		if(B < 0)
		{
			q = (-B - discSqrt) / 2.0f;
		}
		else
		{
			q = (-B + discSqrt) / 2.0f;
		}

		float t0 = q / A;
		float t1 = C / q;

		if(t0 > t1)
		{
			float temp = t0;
			t0 = t1;
			t1 = temp;
		}

		if(t1 < 0)
		{
			
		}
		else if(t0 < 0)
		{
			t = t1;
		}
		else
		{
			t = t0;
		}
	}

	return t;
}

void RunRayTracer(uchar4* dest, const int imageW, const int imageH, const int xThreadsPerBlock, const float4 a_vCameraPosition, const float4 a_vCameraForward, const float4 a_vCameraUp, const float4 a_vCameraRight, const float a_fNearPlaneDistance)
{
	dim3 numThreads(16, 16);
	dim3 numBlocks(80, 45);

	float2 viewSize;

	viewSize = make_float2((float)imageW, (float)imageH);

	RayTracer<<<numBlocks, numThreads>>>(dest, imageW, imageH, a_vCameraPosition, a_vCameraUp, a_vCameraForward, a_vCameraRight, a_fNearPlaneDistance, viewSize);
}