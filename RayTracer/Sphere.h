#pragma once
#include "Object.h"

#include "Vector.h"

class Sphere :
	public Object
{
public:
	Sphere(const Vector &_center, double _radius);
	~Sphere();
	double intersectionDistance(const Ray &ray);

protected:
	Vector center;
	double radius;
};
