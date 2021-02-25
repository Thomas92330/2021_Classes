#include "Vector.h"
#include <math.h>

Vector::Vector(double _x,double _y,double _z)
	:x(_x),y(_y),z(_z)
{
}

Vector::~Vector()
{
}

Vector Vector::operator + (const Vector &v) const
{
	return Vector(x + v.x, y + v.y, z + v.z);
}

Vector Vector::operator - (const Vector &v) const
{
	return Vector(x - v.x, y - v.y, z - v.z);
}

double Vector::dot(const Vector &v) const
{
	return x * v.x + y * v.y + z * v.z;
}

double Vector::norm() const
{
	return sqrt(x*x+y*y+z*z);
}

void Vector::normalize()
{
	double theNorm = norm();
	if (theNorm != 0) {
		x /= theNorm;
		y /= theNorm;
		z /= theNorm;
	}
}

Vector Vector::operator * (double factor) const
{
	return Vector(x*factor,y*factor,z*factor);
}
