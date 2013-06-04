#include "Shape.h"


Sphere::Sphere(Point center, float radius)
	:	mCenter(center), mRadius(radius)
{

}

bool Sphere::Intersect(Ray* ray, HitInfo* hitInfo)
{
	return false;
}