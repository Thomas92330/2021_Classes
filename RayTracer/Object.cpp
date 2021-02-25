#include "Object.h"

Object::Object()
{
}

Object::~Object()
{
}

void Object::setColor(const Color &_color)
{
	color = _color;
}

const Color &Object::getColor() const
{
	return color;
}
