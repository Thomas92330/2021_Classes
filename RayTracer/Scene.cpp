#include "Scene.h"
#include "Bmpfile.h"
#include "Object.h"
#include "Light.h"
#include "Common.h"

Scene::Scene()
{
}

Scene::~Scene()
{
	/*
	// old way
	for (unsigned int i = 0; i < objects.size(); i++) {
		delete objects[i];
	}
	*/
	// C++ 11
	for (Object *object: objects) {
		delete object;
	}
	for (Light *light : lights) {
		delete light;
	}
}

void Scene::addObject(Object *object)
{
	objects.push_back(object);
}

void Scene::addLight(Light *light)
{
	lights.push_back(light);
}

void Scene::render(std::string fileName, unsigned int width, unsigned int height)
{
	Ray ray;
	ray.setOrigin(Vector(0,0,10));
	Color color;
	unsigned char *buffer = new unsigned char[width*height*3];
	for (unsigned int y = 0; y < height; y++) {
		for (unsigned int x = 0; x < width; x++) {
			unsigned int index = (y * width + x) * 3;
			
			Vector pixelPoint = Vector(
				( (double)x / (double)(width-1) ) - 0.5,
				0.5 - ( (double)y / (double)(height-1) ),
				0
			);
			Vector direction = pixelPoint - ray.getOrigin();
			direction.normalize();
			ray.setDirection(direction);

			color = raytrace(ray);

			buffer[index] = (unsigned char) (255 * color.getBlue());
			buffer[index+1] = (unsigned char)(255 * color.getGreen());
			buffer[index+2] = (unsigned char)(255 * color.getRed());
		}
	}
	BMPFile::SaveBmp(fileName, buffer, width, height);
	delete []buffer;
}

Color Scene::raytrace(const Ray &ray)
{
	Color color(0,0,0);

	double nearestDistance;
	Object *impactedObject = findNearestObject(ray,nearestDistance);

	if (impactedObject != NULL) {
		Vector impact = ray.getOrigin() + ray.getDirection() * nearestDistance;

		color = color + localIllumination(impact, impactedObject);
		// reflect : color = color + raytrace(???);
		// refract : color = color + raytrace(???);
	}

	return color;
}

Object *Scene::findNearestObject(const Ray &ray,double &nearestDistance)
{
	Object *nearestObject = NULL;
	nearestDistance = INFINITE;
	for (Object *object : objects) {
		double distance = object->intersectionDistance(ray);
		if (distance < nearestDistance) {
			nearestDistance = distance;
			nearestObject = object;
		}
	}
	return nearestObject;
}

Color Scene::localIllumination(Vector impact, Object *impactedObject)
{
	Color color;
	color = impactedObject->getColor();
	// TODO : for each light : Lambert N.L (+Phong)
	return color;
}
