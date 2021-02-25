#include "Color.h"

Color::Color(double _r, double _g, double _b)
	:r(_r),g(_g),b(_b)
{
}

Color::~Color()
{
}

Color Color::operator + (const Color &color) const
{
	return Color(r + color.r, g + color.g, b + color.b);
}

double Color::getRed() const
{
	return r;
}
double Color::getGreen() const
{
	return g;
}
double Color::getBlue() const
{
	return b;
}
