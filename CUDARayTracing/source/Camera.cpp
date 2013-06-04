#include "Camera.h"

Camera::Camera()
	:	mLocation(Point(0, 10, 0)), mUp(Vector(0, 1, 0)), mFocalLength(0.1), mViewWidth(1), mViewHeight(0.75)
{
	mForward = Point(0, 0, 10) - mLocation;
	mRight = Cross(mForward, mUp);
	mUp = Cross(mRight, mForward);
}

Camera::Camera(Point location, Point lookAt, Vector up, float focalLength, float viewWidth, float viewHeight, int xPixels, int yPixels)
	:	mLocation(location), mUp(up), mFocalLength(focalLength), mViewWidth(viewWidth), mViewHeight(viewHeight), mPixelsX(xPixels), mPixelsY(yPixels)
{
	mForward = lookAt - mLocation;
}

Camera::~Camera()
{

}

//Ray Camera::CreateRayThroughPixel(int x, int y)
//{
//
//}