#pragma once

#include "Light.h"

#include "Vector.h"

class DirectionalLight :
	public Light
{
public:
	DirectionalLight(Vector _direction);
	~DirectionalLight();
protected:
	Vector direction;
};
