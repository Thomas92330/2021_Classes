#pragma once

#include "Light.h"
#include "Vector.h"
class PointLight : public Light
{
public:
	PointLight(Vector _origine);

	~PointLight();

	Vector VectorL(const Vector & point);
protected:
	Vector origine;
};