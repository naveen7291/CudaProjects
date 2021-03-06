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
#include <helper_cuda.h>
#include <cuda_runtime.h>


__device__ float SphereIntersection(float3 rayOrigin, float3 rayDirection, float3 spherePosition, float sphereRadius);
__device__ float QuadraticSolver(float A, float B, float C);
__device__ float4 PointLightContribution(float3 position, float3 normal, float4 color, float3 lightPosition, float3 cameraPosition);
__device__ float4 GetSphereColor(int sphereIndex);
__device__ float3 GetSpherePosition(int sphereIndex);
__device__ float GetSphereRadius(int sphereIndex);

texture<float, 1, cudaReadModeElementType> tex;

__global__ void PostProcessing(uchar4* dest, const int imageW, const int imageH)
{
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y;

	float4 newColor;

	if(ix > 0 && ix < (imageW - 1) && iy > 0 && iy < (imageH - 1))
	{
		newColor = make_float4(0, 0, 0, 0);
		for(int x = -1; x < 2; ++x)
		{
			for(int y = -1; y < 2; ++y)
			{
				int i = imageW * (iy + y) + (ix + x);
				newColor.x += ONE_NINTH * float(dest[i].x);
				newColor.y += ONE_NINTH * float(dest[i].y);
				newColor.z += ONE_NINTH * float(dest[i].z);
			}
		}
		newColor.w = 255;

		int i = imageW * iy + ix;
		dest[i].x = unsigned char(newColor.x);
		dest[i].y = unsigned char(newColor.y);
		dest[i].z = unsigned char(newColor.z);
		dest[i].w = 255;
	}
}

__global__ void RayTracerWithTexture(uchar4* dest, const int imageW, const int imageH, float3 cameraPosition, float3 cameraUp, float3 cameraForward, float3 cameraRight, float nearPlaneDistance, float2 viewSize, int numSpheres)
{
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if(ix >= imageW || iy >= imageH)
	{
		return;
	}

	// Compute the location in the dest array that will be written to
	const int pixelIndex = imageW * iy + ix;
	float4 colorComponents[REFLECTION_DEPTH];
	float4 pixelColor;

	// Compute the center of the near plane. All rays will be computed as an offset from this point
	const float3 lookAt = cameraPosition + cameraForward * nearPlaneDistance;

	// Find where the ray intersects the near plane and create the vector portion of the ray from that
	const float3 rayMidPoint = lookAt + cameraRight * ((float(ix) / float(imageW) - 0.5f) * viewSize.x) + cameraUp * ((float(iy) / float(imageH) - 0.5f) * viewSize.y); 
	float3 ray = normalize(rayMidPoint - cameraPosition);

	const float3 lightPosition = make_float3(0, 30, 25);
	
	float4 sphereColor;
	float3 sphereCenter;
	float3 intersectionPoint;
	float3 intersectionNormal;
	float radius;
	float t = INFINITY;
	float tMin = INFINITY;
	int iClosestSphere;

	// Test the camera ray against all spheres in the scene
	for(int i = 0; i < numSpheres; ++i)
	{
		sphereCenter = GetSpherePosition(i);
		radius = GetSphereRadius(i);

		t = SphereIntersection(cameraPosition, ray, sphereCenter, radius);
		
		// track the closest sphere
		if(t > 0 && t < tMin)
		{
			tMin = t;
			iClosestSphere = i;
		}
	}

	
	for(int i = 0; i < REFLECTION_DEPTH; ++i)
	{
		// If the camera ray intersected with a sphere, compute lighting and reflection
		if(tMin < INFINITY)
		{
			float3 shadowRay;
			float3 intersectedSphereCenter;
			float intersectedSphereRadius;
			int iClosestReflectedSphere;

			sphereColor = GetSphereColor(iClosestSphere);
			sphereCenter = GetSpherePosition(iClosestSphere);

			// Find the point of intersection on the sphere and the normal at that point
			intersectionPoint = cameraPosition + tMin * ray;
			intersectionNormal = normalize(intersectionPoint - sphereCenter);
			shadowRay = normalize(lightPosition - intersectionPoint);
			ray = reflect(ray, intersectionNormal);

			t = INFINITY;
			tMin = INFINITY;
		
			// Test for intersection using a shadow ray from the intersection point to the light source
			for(int j = 0; j < numSpheres; ++j)
			{
				// Don't compare the shadowed sphere against itself
				if(j == iClosestSphere)
				{
					continue;
				}

				intersectedSphereCenter = GetSpherePosition(j);
				intersectedSphereRadius = GetSphereRadius(j);

				t = SphereIntersection(intersectionPoint, shadowRay, intersectedSphereCenter, intersectedSphereRadius);
				if(t > 0 && t < tMin)
				{
					tMin = t;
				}
			}

			// if a sphere was intersected, cast a shadow over the color
			if(tMin < INFINITY)
			{
				colorComponents[i] = sphereColor * AMBIENT_STRENGTH;
				colorComponents[i].w = 1.0f;
			}
			// otherwise calculate lighting
			else
			{
				colorComponents[i] = PointLightContribution(intersectionPoint, intersectionNormal, sphereColor, lightPosition, cameraPosition);
			}

			// Test for intersection using a reflected ray
			for(int j = 0; j < numSpheres; ++j)
			{
				if(j == iClosestSphere)
				{
					continue;
				}

				intersectedSphereCenter = GetSpherePosition(j);
				intersectedSphereRadius = GetSphereRadius(j);

				t = SphereIntersection(intersectionPoint, ray, intersectedSphereCenter, intersectedSphereRadius);
				if(t > 0 && t < tMin)
				{
					tMin = t;
					iClosestReflectedSphere = j;
				}
			}

			// if the reflected ray intersected, store the sphere and start over again
			if(tMin < INFINITY)
			{
				iClosestSphere = iClosestReflectedSphere;
			}
			// otherwise do the reflection calculation
			else
			{
				int j;

				for(j = i; j >= 1; --j)
				{
					colorComponents[j - 1] = colorComponents[j - 1] * 0.70f + colorComponents[j] * 0.30f;
				}

				pixelColor = colorComponents[j];

				break;
			}
		}
		else
		{
			pixelColor = make_float4(BACKGROUND_COLOR);
		}
	}

	dest[pixelIndex] = make_uchar4((unsigned char)(pixelColor.x * 255), (unsigned char)(pixelColor.y * 255), (unsigned char)(pixelColor.z * 255), 255);
}

__device__ float4 GetSphereColor(int sphereIndex)
{
	return make_float4(	tex1D(tex, sphereIndex * SPHERE_NUMFLOATS + SPHERE_COLOR_R), 
						tex1D(tex, sphereIndex * SPHERE_NUMFLOATS + SPHERE_COLOR_G), 
						tex1D(tex, sphereIndex * SPHERE_NUMFLOATS + SPHERE_COLOR_B), 
						tex1D(tex, sphereIndex * SPHERE_NUMFLOATS + SPHERE_COLOR_A));
}

__device__ float3 GetSpherePosition(int sphereIndex)
{
	return make_float3(	tex1D(tex, sphereIndex * SPHERE_NUMFLOATS + SPHERE_POS_X), 
						tex1D(tex, sphereIndex * SPHERE_NUMFLOATS + SPHERE_POS_Y), 
						tex1D(tex, sphereIndex * SPHERE_NUMFLOATS + SPHERE_POS_Z));
}

__device__ float GetSphereRadius(int sphereIndex)
{
	return tex1D(tex, sphereIndex * SPHERE_NUMFLOATS + SPHERE_RADIUS);
}

__device__ float4 PointLightContribution(float3 position, float3 normal, float4 color, float3 lightPosition, float3 cameraPosition)
{
		const float3 lightDirection = normalize(lightPosition - position);
		const float3 halfVector = normalize(lightDirection + normalize(cameraPosition - position));
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

__device__ float SphereIntersection(float3 rayOrigin, float3 rayDirection, float3 spherePosition, float sphereRadius)
{
	// Calculate the three coefficients in the quadratic equation
	const float3 rayOriginMinusSphereCenter = rayOrigin - spherePosition;

	const float A = dot(rayDirection, rayDirection);
	const float B = 2 * dot(rayOriginMinusSphereCenter, rayDirection);
	const float C = dot(rayOriginMinusSphereCenter, rayOriginMinusSphereCenter) - sphereRadius * sphereRadius;

	return QuadraticSolver(A, B, C);
}

__device__ float QuadraticSolver(float A, float B, float C)
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

void RunRayTracerWithTexture(float* sceneData, int sceneSize, uchar4* dest, const int imageW, const int imageH, const int xThreadsPerBlock, const float3 a_vCameraPosition, const float3 a_vCameraForward, const float3 a_vCameraUp, const float3 a_vCameraRight, const float a_fNearPlaneDistance)
{
	int xBlocks, yBlocks;
	xBlocks = imageW / THREAD_COUNT;
	yBlocks = imageH / THREAD_COUNT;

	if(imageW % THREAD_COUNT > 0)
	{
		xBlocks++;
	}
	if(imageH % THREAD_COUNT > 0)
	{
		yBlocks++;
	}

	dim3 numThreads(THREAD_COUNT, THREAD_COUNT);
	dim3 numBlocks(xBlocks, yBlocks);
	float2 viewSize;

	viewSize = make_float2((float)imageW, (float)imageH);
    
	cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc<float>();
 
    cudaArray *cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, sceneSize * SIZEOF_SPHERE, 1));
	checkCudaErrors(cudaMemcpyToArray(cuArray, 0, 0, sceneData, sceneSize * SIZEOF_SPHERE, cudaMemcpyHostToDevice));

	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false;
	
	checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));


	RayTracerWithTexture<<<numBlocks, numThreads>>>(dest, imageW, imageH, a_vCameraPosition, a_vCameraUp, a_vCameraForward, a_vCameraRight, a_fNearPlaneDistance, viewSize, sceneSize);
	//PostProcessing<<<numBlocks, numThreads>>>(dest, imageW, imageH);

	//huge performance decrease
	checkCudaErrors(cudaFreeArray(cuArray));
}
