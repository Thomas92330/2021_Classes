#include "Sphere.h"
#include "Common.h"
#include <Math.h>

Sphere::Sphere(const Vector &_center,double _radius)
	:Object(),center(_center),radius(_radius)
{
}

Sphere::~Sphere()
{
}

double Sphere::intersectionDistance(const Ray &ray) 
{
	Vector CO = ray.getOrigin() - center;
	double b =  2 *(CO.dot(ray.getDirection()));
	double c = (CO.dot(CO)) - radius * radius;
	double disc = b * b - 4 * c ;
	if (disc < 0) {
		return INFINITE;
	}
	double t = (-b - sqrt(disc)) / 2;
	if (t > 0) {
		return t;
	}
	t = (-b + sqrt(disc)) / 2;
	if (t > 0) {
		return t;
	}
	return INFINITE;
}
