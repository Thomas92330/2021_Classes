#pragma once

#include "Color.h"
#include "Ray.h"

class Object
{
public:
	Object();
	virtual ~Object();

	void setColor(const Color &_color);
	const Color &getColor() const;

	virtual double intersectionDistance(const Ray &ray) =0;
protected:
	Color color;
};
