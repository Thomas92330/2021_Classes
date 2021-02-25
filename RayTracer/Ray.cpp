#include "Ray.h"

Ray::Ray()
	:origin(), direction()
{
}

Ray::Ray(const Vector &_origin, const Vector &_direction)
	:origin(_origin),direction(_direction)
{
}

Ray::~Ray()
{
}

void Ray::setOrigin(const Vector &_origin)
{
	origin = _origin;
}

void Ray::setDirection(const Vector &_direction)
{
	direction = _direction;
}

const Vector &Ray::getOrigin() const
{
	return origin;
}

const Vector &Ray::getDirection() const
{
	return direction;
}
