#include <stdio.h>
#include <stdlib.h>

#include "RayTracer.h"
#include "geometry.h"

// OpenGL Graphics includes
#include <GL/glew.h>
#ifdef _WIN32
#include <GL/wglew.h>
#endif
#include <GL/freeglut.h>


// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>
//#include <helper_cuda_gl.h>
//#include <rendercheck_gl.h>

void RenderImage();
void UpdateInputs();
void ComputeFPS();

void Display();
void Reshape(int w, int h);
void ProcessKeyboard(unsigned char k, int, int);
void ProcessKeyboardUp(unsigned char k, int, int);
void ProcessMouseClick(int button, int state, int x, int y);
void ProcessMouseMove(int x, int y);
void TimerEvent(int value);

void InitializeOpenGL(int* argc, char** argv);
void InitializeOpenGLBuffers(int w, int h);
GLuint compileASMShader(GLenum program_type, const char *code);
void CleanUp();

// OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange

uchar4 *h_Source = NULL;
uchar4 *d_Dest = NULL;

StopWatchInterface *hTimer = NULL;
const int frameCheckNumber = 60;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 15;       // FPS limit for sampling
unsigned int frameCount = 0;

// Tweakable values
float4 g_vCameraLocation;
float4 g_vCameraForward;
float4 g_vCameraRight;
float4 g_vCameraUp;
float g_fNearPlaneDistance;

// Input flags
bool g_aInputFlags[NUMBER_OF_INPUTS];
bool g_bMouseLeftDown;
bool g_bMouseRightDown;
int g_nLastMouseX;
int g_nLastMouseY;

int main(int argc, char** argv)
{
	InitializeOpenGL(&argc, argv);

	sdkCreateTimer(&hTimer);
	sdkStartTimer(&hTimer);

	atexit(CleanUp);

	glutMainLoop();

	cudaDeviceReset();
}

void RenderImage()
{
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_Dest, &num_bytes, cuda_pbo_resource));

	RunRayTracer(d_Dest, WINDOW_WIDTH, WINDOW_HEIGHT, 0, g_vCameraLocation, g_vCameraForward, g_vCameraUp, g_vCameraRight, g_fNearPlaneDistance);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

void Display()
{
	sdkStartTimer(&hTimer);

	UpdateInputs();

    RenderImage();

    // load texture from PBO
    //  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    //  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // fragment program is required to display floating point texture
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0.0f, 1.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_FRAGMENT_PROGRAM_ARB);

	sdkStopTimer(&hTimer);
    glutSwapBuffers();

	ComputeFPS();
}

void UpdateInputs()
{
	if(g_aInputFlags['w'])
	{
		g_vCameraLocation += g_vCameraForward * CAMERA_MOVEMENT_DELTA;
	}
	if(g_aInputFlags['s'])
	{
		g_vCameraLocation -= g_vCameraForward * CAMERA_MOVEMENT_DELTA;
	}
	if(g_aInputFlags['a'])
	{
		g_vCameraLocation -= g_vCameraRight * CAMERA_MOVEMENT_DELTA;
	}
	if(g_aInputFlags['d'])
	{
		g_vCameraLocation += g_vCameraRight * CAMERA_MOVEMENT_DELTA;
	}
	if(g_aInputFlags['o'])
	{
		g_vCameraLocation -= g_vCameraUp * CAMERA_MOVEMENT_DELTA;
	}
	if(g_aInputFlags['p'])
	{
		g_vCameraLocation += g_vCameraUp * CAMERA_MOVEMENT_DELTA;
	}
	if(g_aInputFlags['-'])
	{
		g_fNearPlaneDistance -= NEAR_PLANE_MOVEMENT_DELTA;
	}
	if(g_aInputFlags['='])
	{
		g_fNearPlaneDistance += NEAR_PLANE_MOVEMENT_DELTA;
	}
}

void ComputeFPS()
{
	frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&hTimer) / 1000.f);
        sprintf(fps, "CUDA Ray Tracer %3.1f fps", ifps);
        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(1.f, (float)ifps);
        sdkResetTimer(&hTimer);
    }
}

void Reshape(int w, int h)
{
    //glViewport(0, 0, w, h);
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    //InitializeOpenGLBuffers(w, h);
	InitializeOpenGLBuffers(WINDOW_WIDTH, WINDOW_HEIGHT);
}

void ProcessKeyboard(unsigned char k, int, int)
{
	switch (k)
    {
        case '\033':
        case 'q':
        case 'Q':
            printf("Shutting down...\n");
            exit(EXIT_SUCCESS);
            break;

		default:
			g_aInputFlags[k] = true;
   }

   printf("Camera Location: (%f, %f, %f, %f), NearPlaneDistance: %f\n", g_vCameraLocation.x, g_vCameraLocation.y, g_vCameraLocation.z, g_vCameraLocation.w, g_fNearPlaneDistance);
}

void ProcessKeyboardUp(unsigned char k, int, int)
{
	g_aInputFlags[k] = false;
}

void ProcessMouseClick(int button, int state, int x, int y)
{
	if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		g_bMouseLeftDown = true;
	}
	if(button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{
		g_bMouseLeftDown = false;
	}
	if(button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
	{
		g_bMouseRightDown = true;
	}
	if(button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
	{
		g_bMouseRightDown = false;
	}

	g_nLastMouseX = x;
	g_nLastMouseY = y;
}

void ProcessMouseMove(int x, int y)
{
	printf("%d %d\n", x, y);

	float xAngle = (x - g_nLastMouseX) / 90.0f;
	float yAngle = (y - g_nLastMouseY) / 90.0f;

	clamp(xAngle, 0.0f, 0.3f);
	clamp(yAngle, 0.0f, 0.3f);

	g_vCameraForward.z = (g_vCameraForward.z * cos((float)xAngle) - (g_vCameraForward.x * sin((float)xAngle)));
	g_vCameraForward.x = (g_vCameraForward.x * cos((float)xAngle) + (g_vCameraForward.z * sin((float)xAngle)));

	//normalize(g_vCameraForward);
	//normalize(g_vCameraUp);

	printf("Forward: %f %f %f\n", g_vCameraForward.x, g_vCameraForward.y, g_vCameraForward.z);
	printf("Up: %f %f %f\n", g_vCameraUp.x, g_vCameraUp.y, g_vCameraUp.z);
	printf("Right: %f %f %f\n", g_vCameraRight.x, g_vCameraRight.y, g_vCameraRight.z);

	g_vCameraRight = normalize(CRTUtil::cross(g_vCameraUp, g_vCameraForward));
	g_vCameraForward = normalize(CRTUtil::cross(g_vCameraRight, g_vCameraUp));

	g_nLastMouseX = x;
	g_nLastMouseY = y;
}

void TimerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, TimerEvent, 0);
}

void InitializeOpenGL(int* argc, char** argv)
{
	printf("Initializing GLUT");

    glutInit(argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitWindowPosition(WINDOW_POS_X, WINDOW_POS_Y);
    glutCreateWindow(argv[0]);

	glutDisplayFunc(Display);
	glutKeyboardFunc(ProcessKeyboard);
	glutKeyboardUpFunc(ProcessKeyboardUp);
	glutMouseFunc(ProcessMouseClick);
	glutMotionFunc(ProcessMouseMove);
	glutReshapeFunc(Reshape);
    glutTimerFunc(REFRESH_DELAY, TimerEvent, 0);

    printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));

    if (!glewIsSupported("GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
        fprintf(stderr, "This sample requires:\n");
        fprintf(stderr, "  OpenGL version 1.5\n");
        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
        exit(EXIT_SUCCESS);
    }

    printf("OpenGL window created.\n");	

	//Initialize tweakable values
	g_vCameraLocation = make_float4(CAMERA_LOCATION);
	g_vCameraForward = make_float4(CAMERA_FORWARD);
	g_vCameraUp = make_float4(CAMERA_UP);
	g_vCameraRight = make_float4(CAMERA_RIGHT);
	g_fNearPlaneDistance = NEAR_PLANE_DISTANCE;

	for(int i = 0; i < NUMBER_OF_INPUTS; ++i)
	{
		g_aInputFlags[i] = false;
	}

	g_bMouseLeftDown = false;
	g_bMouseRightDown = false;
}

// gl_Shader for displaying floating-point texture
static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

void InitializeOpenGLBuffers(int w, int h)
{
   // delete old buffers
    if (h_Source)
    {
        free(h_Source);
        h_Source = 0;
    }

    if (gl_Tex)
    {
        glDeleteTextures(1, &gl_Tex);
        gl_Tex = 0;
    }

    if (gl_PBO)
    {
        //DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &gl_PBO);
        gl_PBO = 0;
    }

    // check for minimized window
    if ((w==0) && (h==0))
    {
        return;
    }

    // allocate new buffers
    h_Source = (uchar4 *)malloc(w * h * 4);

    printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Source);
    printf("Texture created.\n");

    printf("Creating PBO...\n");
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Source, GL_STREAM_COPY);
    //While a PBO is registered to CUDA, it can't be used
    //as the destination for OpenGL drawing calls.
    //But in our particular case OpenGL is only used
    //to display the content of the PBO, specified by CUDA kernels,
    //so we need to register/unregister it only once.

    // DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(gl_PBO) );
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO,
                                                 cudaGraphicsMapFlagsWriteDiscard));
    printf("PBO created.\n");

    // load shader program
    gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1)
    {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }

    return program_id;
}

void CleanUp()
{
   if (h_Source)
    {
        free(h_Source);
        h_Source = 0;
    }

    sdkStopTimer(&hTimer);
    sdkDeleteTimer(&hTimer);

    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glDeleteBuffers(1, &gl_PBO);
    glDeleteTextures(1, &gl_Tex);
    glDeleteProgramsARB(1, &gl_Shader);
    exit(EXIT_SUCCESS);
}
