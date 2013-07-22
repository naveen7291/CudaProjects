#ifndef _CUDA_RAY_TRACER_h_
#define _CUDA_RAY_TRACER_h_

// OPENGL CONFIGURATION

// Window Settings
//#define FULLSCREEN
#define WINDOW_HEIGHT		720
#define WINDOW_WIDTH		1280
#define WINDOW_POS_X		0
#define WINDOW_POS_Y		0

// Timer settings
#define REFRESH_DELAY 10

// ALGORITHM PARAMETERS
#define REFLECTION_DEPTH	8
#define NUMBER_OF_SPHERES	3
#define THREAD_COUNT		16

// MATHEMATICAL CONSTANTS
#define FLT_MIN			1.175494351e-38F
#define FLT_MAX			3.402823466e+38F
#define INFINITY		FLT_MAX
#define ONE_NINTH		1.0f/9.0f

// INITIAL CAMERA VALUES
#define CAMERA_LOCATION				0, 0, -1000
#define CAMERA_FORWARD				0, 0, 1
#define CAMERA_UP					0, 1, 0
#define CAMERA_RIGHT				1, 0, 0
#define NEAR_PLANE_DISTANCE			1500.0f
#define CAMERA_MOVEMENT_DELTA		2.0f
#define NEAR_PLANE_MOVEMENT_DELTA	10.0f

// LIGHT CONSTANTS
#define AMBIENT_STRENGTH			0.35f
#define BACKGROUND_COLOR			0.2f, 0.2f, 0.4f, 1.0f

// INPUT CONSTANTS
#define NUMBER_OF_INPUTS			256

// MEMORY CONSTANTS
#define SPHERE_NUMFLOATS			8
#define SIZEOF_SPHERE				sizeof(float) * SPHERE_NUMFLOATS
#define SPHERE_COLOR_R				0
#define SPHERE_COLOR_G				1
#define SPHERE_COLOR_B				2
#define SPHERE_COLOR_A				3
#define SPHERE_POS_X				4
#define SPHERE_POS_Y				5
#define SPHERE_POS_Z				6
#define SPHERE_RADIUS				7


#include <vector_types.h>
#include <stdlib.h>
#include "LinearAlgebraUtil.h"

extern "C" void RunRayTracer(uchar4* dest, const int imageW, const int imageH, const int xThreadsPerBlock, const float3 a_vCameraPosition, const float3 cameraForward, const float3 cameraUp, const float3 cameraRight, const float a_fNearPlaneDistance);
extern "C" void RunRayTracerWithTexture(float* sceneData, int sceneSize, uchar4* dest, const int imageW, const int imageH, const int xThreadsPerBlock, const float3 a_vCameraPosition, const float3 cameraForward, const float3 cameraUp, const float3 cameraRight, const float a_fNearPlaneDistance);

#endif