// Raytracer.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//

#include <iostream>

#include "Scene.h"
#include "Sphere.h"

int main()
{
	Scene scene;

	Sphere *sphere = new Sphere(Vector(0,0,0),0.1);
	sphere->setColor(Color(1,0,0));
	scene.addObject(sphere);

	scene.render("test.bmp",1024,768);

	return 0;
}
