#include "geometry.h"

class HitInfo
{
public:
	Point HitLocation;
	Normal HitNormal;
};

class Shape
{
public:
	//Shape();
	//virtual ~Shape();

	virtual bool Intersect(Ray* ray, HitInfo* hitInfo) = 0;

};

class Sphere : public Shape
{
public:
	Sphere(Point center, float radius);
	~Sphere();

	virtual bool Intersect(Ray* ray, HitInfo* hitInfo);

private:
	Point mCenter;
	float mRadius;
};