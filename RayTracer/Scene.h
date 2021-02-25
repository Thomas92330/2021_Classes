#pragma once

#include <string>
#include <vector>

#include "Color.h"
#include "Ray.h"

class Object;
class Light;

class Scene
{
public:
	Scene();
	~Scene();
	void addObject(Object *object);
	void addLight(Light *light);
	void render(std::string fileName,unsigned int width,unsigned int height);

protected:
	Color raytrace(const Ray &ray);
	Object *findNearestObject(const Ray &ray, double &nearestDistance);
	Color localIllumination(Vector impact, Object *impactedObject);

	std::vector<Object*> objects;
	std::vector<Light*> lights;
};
