#include "geometry.h"

class Camera
{
public:
	Camera();
	Camera(Point location, Point lookAt, Vector up, float focalLength, float viewWidth, float viewHeight, int pixelsX, int pixelsY);
	~Camera();

	Ray CreateRayThroughPixel(int x, int y);

private:
	Point mLocation;

	Vector mUp;
	Vector mForward;
	Vector mRight;

	float mFocalLength;
	float mViewWidth;
	float mViewHeight;

	int mPixelsX;
	int mPixelsY;
};