#pragma once
class Vector
{
public:
	Vector(double _x=0, double _y=0, double _z=0);
	~Vector();

	Vector operator + (const Vector &v) const;
	Vector operator - (const Vector &v) const;
	Vector operator * (double factor) const;
	double dot(const Vector &v) const;
	double norm() const;
	void normalize();
protected:
	double x, y, z;
};
