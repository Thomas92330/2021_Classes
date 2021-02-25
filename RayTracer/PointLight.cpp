#include "PointLight.h"

PointLight::PointLight(Vector _origine)
	:Light(), origine(_origine)
{
}

PointLight::~PointLight()
{
}


Vector PointLight::VectorL(const Vector& point)
{
	Vector vectorL = (point - origine);
	vectorL.normalize();

	return vectorL;
}
