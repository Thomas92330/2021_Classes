#pragma once

#include "Vector.h"

class Ray
{
public:
	Ray();
	Ray(const Vector &_origin,const Vector &_direction);
	~Ray();

	void setOrigin(const Vector &_origin);
	void setDirection(const Vector &_direction);

	const Vector &getOrigin() const;
	const Vector &getDirection() const;

protected:
	Vector origin;
	Vector direction;
};
