#include <GL/glut.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

VideoCapture capture;
Mat frame;

void keyboard(unsigned char key, int x, int y)
{

}

void drag(int x, int y)
{

}

void mouse(int button, int state, int x, int y)
{

}

void init() {
	GLfloat yellow[] = { 1.0, 1.0, 0.0, 1.0 };
	GLfloat green[] = { 0.0, 1.0, 0.0, 1.0 };
	GLfloat white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	GLfloat direction[] = { 1.0, 1.0, 1.0, 0.0 };

	// specify material parameters for the lighting model
	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, green);
	glMaterialfv(GL_FRONT, GL_SPECULAR, white);
	glMaterialf(GL_FRONT, GL_SHININESS, 50);

	// set light source parameters
	glLightfv(GL_LIGHT0, GL_AMBIENT, black);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, yellow);
	glLightfv(GL_LIGHT0, GL_SPECULAR, white);
	glLightfv(GL_LIGHT0, GL_POSITION, direction);

	glEnable(GL_LIGHTING);   // enable light
	glEnable(GL_LIGHT0);     // turn LIGHT0 on
	glEnable(GL_DEPTH_TEST); // consider depth
}

void matToTexture(Mat image)
{
	GLuint texture_id;
	if (image.empty())
		cerr << "Error : couldn't read the image ..." << endl;
	else
	{
		glPushMatrix(); // push the current matrix down the stack 
		glDisable(GL_LIGHTING); // disable lighting to preserve camera image light 
		glDisable(GL_DEPTH_TEST); // disable depth comparisons 
		glViewport(0, 0, image.cols, image.rows); // specify window coordinates rectangle
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, image.cols, 0, image.rows, -1, 1);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glGenTextures(1, &texture_id); // generate a texture
		// bind the texture to a texturing target (2D texture)
		glBindTexture(GL_TEXTURE_2D, texture_id);

		// set texture parameters		
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // texture minifying
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // texture magnification
		
		Mat rgbImage, flipedImage;		
		cvtColor(image, rgbImage, COLOR_BGR2RGB); // transform OpenCV BGR image to RGB (OpenGL format)
		flip(rgbImage, flipedImage, 0); // flip around x-axis

		// define a texture with the image image
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, flipedImage.cols, flipedImage.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, flipedImage.data);
				
		glEnable(GL_TEXTURE_2D); // define a texture
		// define a quadrilateral for texture display 
		glBegin(GL_QUADS);
		glTexCoord2f(0, 0); glVertex3f(0, 0, 0);
		glTexCoord2f(0, 1); glVertex3f(0, flipedImage.rows, 0);
		glTexCoord2f(1, 1); glVertex3f(flipedImage.cols, flipedImage.rows, 0);
		glTexCoord2f(1, 0); glVertex3f(flipedImage.cols, 0, 0);
		glEnd();
		glDisable(GL_TEXTURE_2D);
		glEnable(GL_LIGHTING); // enable lighting again
		glEnable(GL_DEPTH_TEST); // depth comparisons and update the depth buffer
		glDeleteTextures(1, &texture_id); // delete the texture to free memory 
		glPopMatrix(); // pop the current matrix stack
	}
}

void display()
{	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear color and depth buffer at each display call

	capture >> frame;
	if (frame.empty()) return;

	//matToTexture(frame); // transform the camera frame into a GL texture

	glPushMatrix(); // push the current matrix down the stack  
	
	glViewport(0, 0, frame.cols, frame.rows); // define the GL viewport with the same frame size
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	float zNear = 1, zFar = 100;
	glFrustum(-1, 1, -1, 1, zNear, zFar); // define the projection view field
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 0, 0, 0, 0, 1, 0, 1, 0); // transform the modelview into a right-handed coordinate system

	// add translations
	glTranslatef(0, 0, 10);

	// add rotations
	
	// add scale

	// load 3D model
	glutSolidTeapot(1); // draw a teapot

	glPopMatrix(); // pop the current matrix stack

	glFlush(); // execute commands
	glutPostRedisplay(); // redisplay
}

int main(int argc, char** argv)
{
	capture.open("video/videoTuto6.mp4");
	if (!capture.isOpened())
	{
		cerr << "Couldn't open the video ..." << endl;
		return 1;
	}

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(640, 480);
	glutCreateWindow("3D AR Overlay");
	init();
	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(drag);
	glutMainLoop();

	return 0;
}