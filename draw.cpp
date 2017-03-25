#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <iostream>
#include <vector>
#include <fstream>

#ifdef _WIN32
// #include <windows.h>
#include <direct.h>	//function mkdir
#include <io.h>	// function access
#define access _access
#define mkdir _mkdir
#else
#include <sys/stat.h>
#include <sys/types.h>	//function mkdir
#include <unistd.h>	// function access
#endif

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int image_width, image_height;
int winWidth, winHeight;
float wscale = 0.5, hscale = 0.5;	// scale between image size and window size
int tempX, tempY;	// current mouse position
int rflag = 0;	// 1, start drawing rectangle
int targetnum = 0;	// target number in current image
int pflag = 0;	// 1, start drawing polygon
int polygonnum = 0;	// polygon number in current image

int mode = 0;	//0,rectangle mode;  1,polygon mode

int boxtype = 0;	// 0,rectangular; 1,parallelogram TODO

enum recttype { LEFT = 1, RIGHT, FORWARD, LEFT_FORWARD, RIGHT_FORWARD, TURN_ROUND, RT_NUM };
enum polygontype { PARKING = 1, SOLID_WHITE, DASHED_WHITE, SOLID_YELLOW, DASHED_YELLOW, DOUBLE_YELLOW, PT_NUM };
int curtype = 0;	//default type 0

vector<Vec3f> rect_colors;	// color vectors of different rect target
vector<Vec3f> polygon_colors;	// color vectors of different polygon

// target node
struct RectfNode {
	int x1;
	int y1;
	int x2;
	int y2;
};

std::vector<RectfNode> targets;
std::vector<int> targetType;

std::vector<vector<Point2i> > polygons;
std::vector<int> polygonType;

GLuint texture;	// texture of current image
// string imgdir = "/home/tony/Documents/matconvnet-master/matconvnet-fcn-master/data/voc11/JPEGImages/";	// images directory
// string imgdir = "images/";
string imgdir, labeldir, targetdir, segdir;
vector<cv::String> imgpaths;	// images path list
vector<string> targetFiles;	// target files list
int imgsLen;	// images length
int imgnum = 0;	// current image number, default 0

char winName[20];	// window title

Mat instruction(450, 550, CV_8UC3, Scalar::all(255));	// show instruction

void saveLabel(){

	string targetfile = targetdir + targetFiles[imgnum] + ".txt";
	string segImage = segdir + targetFiles[imgnum] + "_seg.png";
	string segfile = segdir + targetFiles[imgnum] + "_points.txt";

	ofstream tf(targetfile.c_str());
	if (!tf.is_open()){
		cout << "fail to open target file: " << targetfile << endl;
		return ;
	}
	for(int i = 0; i < targetnum; i++){
		tf << targetType[i] << " ";
		tf << targets[i].x1 << " " << image_height - targets[i].y1 << " ";
		tf << targets[i].x2 - targets[i].x1 << " " << -(targets[i].y2 - targets[i].y1) << endl;
	}
	tf.close();

	Mat segImg(image_height, image_width, CV_8UC1, Scalar(0));
	ofstream sf(segfile.c_str());
	if (!sf.is_open()){
		cout << "fail to open segment file: " << segfile << endl;
		return ;
	}
	for (int j = 0; j < polygonnum; j++){
		fillConvexPoly(segImg, polygons[j], Scalar(polygonType[j]));
		sf << polygonType[j] << " " << polygons[j].size() << " ";
		for (int k = 0; k < polygons[j].size(); k++){
			sf << polygons[j][k].x << " " << polygons[j][k].y << " ";
		}
		sf << endl;
	}
	sf.close();
	imwrite(segImage, segImg);
}

/*
 * initialize image lists
 */
void initial(){

	FileStorage fs("config.yml", FileStorage::READ);
	fs["dataPath"] >> imgdir;
	fs.release();
	cout << "data path: " << imgdir << endl;

	string labeldir = imgdir + "../label/";
	string targetdir = labeldir + "target/";
	string segdir = labeldir + "segmentation/";
	cout << "label path: " << labeldir << endl;
	if (access(labeldir.c_str(), 0) != 0){	// create labeldir, targetdir and segdir if not existed
		#ifdef _WIN32
		int status = mkdir(labeldir.c_str());
		int status1 = mkdir(targetdir.c_str());
		int status2 = mkdir(segdir.c_str());
		#else
		int status = mkdir(labeldir.c_str(), S_IRWXU);
		int status1 = mkdir(targetdir.c_str(), S_IRWXU);
		int status2 = mkdir(segdir.c_str(), S_IRWXU);
		#endif
		if (status != 0 || status1 != 0 || status2 != 0){
			cout << "fail to create directory!" << endl;
		}
	}

	// get all the image paths in 'imgdir' and save it in 'imgpaths'
	glob(imgdir, imgpaths, false);

	size_t st = imgdir .length();
	for (vector<cv::String>::iterator it = imgpaths.begin(); it != imgpaths.end(); it++){

		size_t en = (*it).find(".");
		string ext = (*it).substr(en + 1);	// getfile extension
		if(!strcmp(ext.c_str(), "bmp") || !strcmp(ext.c_str(), "jpg") || !strcmp(ext.c_str(), "png")){

			string imgname = (*it).substr(st, en - st);
			// string targetFilename = imgname + ".xml";
			targetFiles.push_back(imgname);
		}
		else {
			it = imgpaths.erase(it);	// delete non-image file
			it--;
		}

	}

	imgsLen = imgpaths.size();

	Mat img = imread(imgpaths[0]);
	if(!img.data){
		cout << "fail to load image, press any key to stop..." << endl;
		getchar();
		exit(0);
	}

	image_width = img.cols;
	image_height = img.rows;
	if(image_width < 500 || image_height < 500){
		wscale = 1.0;
		hscale = 1.0;
	}

	rect_colors.push_back(Vec3f(0.0f, 0.0f, 0.0f));	// default type color
	rect_colors.push_back(Vec3f(0.5f, 0.0f, 0.0f));
	rect_colors.push_back(Vec3f(0.0f, 0.5f, 0.0f));
	rect_colors.push_back(Vec3f(0.0f, 0.0f, 0.5f));
	rect_colors.push_back(Vec3f(0.5f, 0.0f, 0.5f));
	rect_colors.push_back(Vec3f(0.0f, 0.5f, 0.5f));
	rect_colors.push_back(Vec3f(0.5f, 0.5f, 0.0f));


	polygon_colors.push_back(Vec3f(0.0f, 0.0f, 0.0f));	// default type color
	polygon_colors.push_back(Vec3f(1.0f, 0.0f, 0.0f));
	polygon_colors.push_back(Vec3f(0.0f, 1.0f, 0.0f));
	polygon_colors.push_back(Vec3f(0.0f, 0.0f, 1.0f));
	polygon_colors.push_back(Vec3f(1.0f, 0.0f, 1.0f));
	polygon_colors.push_back(Vec3f(1.0f, 1.0f, 0.0f));
	polygon_colors.push_back(Vec3f(0.0f, 1.0f, 1.0f));

	putText(instruction, "Rect Target:", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
	line(instruction, Point(10, 50), Point(60, 50), Scalar(0, 0, 128), 3);
	putText(instruction, "1 Left", Point(70, 55), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
	line(instruction, Point(10, 80), Point(60, 80), Scalar(0, 128, 0), 3);
	putText(instruction, "2 Right", Point(70, 85), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
	line(instruction, Point(10, 110), Point(60, 110), Scalar(128, 0, 0), 3);
	putText(instruction, "3 Forward", Point(70, 115), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
	line(instruction, Point(10, 140), Point(60, 140), Scalar(128, 0, 128), 3);
	putText(instruction, "4 Left-Forward", Point(70, 145), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
	line(instruction, Point(10, 170), Point(60, 170), Scalar(128, 128, 0), 3);
	putText(instruction, "5 Right-Forward", Point(70, 175), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
	line(instruction, Point(10, 200), Point(60, 200), Scalar(0, 128, 128), 3);
	putText(instruction, "6 TURN-ROUND", Point(70, 205), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
	

	putText(instruction, "Polygon:", Point(10, 220 + 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
	line(instruction, Point(10, 250 + 30), Point(60, 250 + 30), Scalar(0, 0, 255), 1);
	putText(instruction, "1 Parking", Point(70, 255 + 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
	line(instruction, Point(10, 280 + 30), Point(60, 280 + 30), Scalar(0, 255, 0), 1);
	putText(instruction, "2 Solid-White", Point(70, 285 + 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
	line(instruction, Point(10, 310 + 30), Point(60, 310 + 30), Scalar(255, 0, 0), 1);
	putText(instruction, "3 Dashed-White", Point(70, 315 + 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
	line(instruction, Point(10, 340 + 30), Point(60, 340 + 30), Scalar(255, 0, 255), 1);
	putText(instruction, "4 Solid-Yellow", Point(70, 345 + 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
	line(instruction, Point(10, 370 + 30), Point(60, 370 + 30), Scalar(0, 255, 255), 1);
	putText(instruction, "5 Dashed-Yellow", Point(70, 375 + 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
	line(instruction, Point(10, 400 + 30), Point(60, 400 + 30), Scalar(255, 255, 0), 1);
	putText(instruction, "6 Double-Yelllow", Point(70, 405 + 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));

	putText(instruction, "Key to press:", Point(250, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0));
	putText(instruction, "n(N): next frame", Point(250, 70), FONT_HERSHEY_SIMPLEX, 0.4, Scalar::all(0));
	putText(instruction, "p(P): prev frame", Point(250, 120), FONT_HERSHEY_SIMPLEX, 0.4, Scalar::all(0));
	putText(instruction, "c(C): change mode, default: Rect", Point(250, 170), FONT_HERSHEY_SIMPLEX, 0.4, Scalar::all(0));
	putText(instruction, "d(D): delete last target or polygon", Point(250, 220), FONT_HERSHEY_SIMPLEX, 0.4, Scalar::all(0));
	putText(instruction, "r(R): reset all targets and polygons to 0", Point(250, 270), FONT_HERSHEY_SIMPLEX, 0.4, Scalar::all(0));
	putText(instruction, "s(S): save targets and polygons", Point(250, 320), FONT_HERSHEY_SIMPLEX, 0.4, Scalar::all(0));

	imshow("instruction", instruction);
	waitKey(10);
}

/*
 * load texture from image file
 */
GLuint LoadTexture(const char * filename){

	GLuint curtexture;
	unsigned char * data;

	//The following code will read in our RAW file
	//At first, I read data directly read data from file, but in this way the image displayed is mislocated, 
	//so I find another way, to avoid this, copy the image data read from opencv
	
	// FILE * file = fopen(filename, "rb");
	// if (file == NULL) return 0;
	// data = (unsigned char *)malloc(width * height * 3);
	// fread(data, width * height * 3, 1, file);
	// fclose(file);
	
	Mat img = imread(filename);
	if(!img.data){
		cout << "fail to load image!" << endl;
		return 0;
	}

	flip(img, img, 0);	// have no idea why image need to flip
	cvtColor(img, img, CV_RGB2BGR);

	// if image size is not always the same, uncomment the following code
	image_width = img.cols;
	image_height = img.rows;
	if(image_width < 500 || image_height < 500){
		wscale = 1.0;
		hscale = 1.0;
	}
	glutReshapeWindow(image_width * wscale, image_height * hscale);

	data = (unsigned char *)malloc(image_width * image_height * 3);
	memcpy(data, img.data, image_width * image_height * 3);

	glGenTextures(1, &curtexture);
	glBindTexture(GL_TEXTURE_2D, curtexture);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	// build texture mipmaps
	gluBuild2DMipmaps(GL_TEXTURE_2D, 3, image_width, image_height, GL_RGB, GL_UNSIGNED_BYTE, data);

	free(data);

	return curtexture; //return whether it was successful

}	

/*
 * draw backround
 */
void cube() {
	//initial the color for textured geometry John
	glColor4f(1, 1, 1, 1);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture); //bind the texture

	glBegin(GL_QUADS);//should be the same size as the clipping volumn John
	glTexCoord2d(0.0, 0.0); glVertex3d(0, 0, 0.0);
	glTexCoord2d(1.0, 0.0); glVertex3d(winWidth, 0, 0.0); 
	glTexCoord2d(1.0, 1.0); glVertex3d(winWidth, winHeight, 0.0);
	glTexCoord2d(0.0, 1.0); glVertex3d(0, winHeight, 0.0);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}

/*
 * load first image texture
 */
void initialTexture(){

	//should always enable this John
	glEnable(GL_DEPTH_TEST);

	texture = LoadTexture(imgpaths[imgnum].c_str());

	// show current image number and images length
	sprintf(winName, "image: %d/%d", imgnum+1, imgsLen);
	glutSetWindowTitle(winName);
}

/*
 * window reshape response function
 */
void ChangeSize(int w, int h){

	winWidth = w;
	winHeight = h;
	wscale = (float)w / (float)image_width;
	hscale = (float)h / (float)image_height;
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//the clipping volume is set as x 0-w y 0-h z 1--1 John
	glOrtho(0., (GLfloat)w, 0, (GLfloat)h, 0.0, 2.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}

/*
 * anything you want to display in the window
 */
void Display(void){
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glLineWidth(3);
	int i = 0;
	// draw rectangle
	for (; i < targetnum; i++){
		glColor3f(rect_colors[targetType[i]][0], rect_colors[targetType[i]][1], rect_colors[targetType[i]][2]);
		glRecti(targets[i].x1 * wscale, targets[i].y1 * hscale, targets[i].x2 * wscale, targets[i].y2 * hscale);

	}
	if (mode == 0 && rflag == 1)
    {
		glColor3f(rect_colors[targetType[i]][0], rect_colors[targetType[i]][1], rect_colors[targetType[i]][2]);
		glRecti(targets[i].x1 * wscale, targets[i].y1 * hscale, tempX, tempY);
	}

	// draw polygon
	int j = 0;
	glLineWidth(1);
	for (; j < polygonnum; j++){
		glColor3f(polygon_colors[polygonType[j]][0], polygon_colors[polygonType[j]][1], polygon_colors[polygonType[j]][2]);
		glBegin(GL_POLYGON);
		for (int k = 0; k < polygons[j].size(); k++){
			glVertex2i(polygons[j][k].x * wscale, polygons[j][k].y * hscale);  
		}
		glEnd();
	}
	if (mode == 1 && pflag == 1){
		glColor3f(polygon_colors[polygonType[j]][0], polygon_colors[polygonType[j]][1], polygon_colors[polygonType[j]][2]);
		int k = 1;
		for (; k < polygons[polygonnum].size(); k++){
			glBegin(GL_LINES);  
				glVertex2i(polygons[polygonnum][k - 1].x * wscale, polygons[polygonnum][k - 1].y * hscale);  
				glVertex2i(polygons[polygonnum][k].x * wscale, polygons[polygonnum][k].y * hscale);  
			glEnd();
		}
		glBegin(GL_LINES);  
			glVertex2i(polygons[polygonnum][k - 1].x * wscale, polygons[polygonnum][k - 1].y * hscale);  
			glVertex2i(tempX, tempY);  
		glEnd();
	}
	
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); 
	
	cube();//draw the background later John
	glutSwapBuffers();
}


//mouse click response function
void MousePlot(GLint button, GLint action, GLint xMouse, GLint yMouse){
	//left button doen
	if (button == GLUT_LEFT_BUTTON && action == GLUT_DOWN){
		if (mode == 0 && rflag == 0){
			rflag = 1;
			RectfNode rn;
			rn.x1 = xMouse / wscale;
			rn.y1 = (winHeight - yMouse) / hscale;
			// cout << xMouse / wscale << " " << yMouse / hscale;

			targets.push_back(rn);
			targetType.push_back(curtype);

		}
		else if (mode == 0 && rflag == 1) {
			rflag = 0;
			targets[targetnum].x2 = xMouse / wscale;
			targets[targetnum].y2 = (winHeight - yMouse) /hscale;
			// cout << "  " << xMouse / wscale << "  " << yMouse / hscale << endl;

			targetnum++;
			glutPostRedisplay();
		}
		else if (mode == 1 && pflag == 0){
			pflag = 1;
			Point2i ps;	// start point
			ps.x = xMouse / wscale;
			ps.y = (winHeight - yMouse) / hscale;

			vector<Point2i> points;
			points.push_back(ps);
			polygons.push_back(points);
			polygonType.push_back(curtype);
		}
		else if (mode == 1 && pflag == 1){
			Point2i pm;
			pm.x = xMouse / wscale;
			pm.y = (winHeight - yMouse) / hscale;

			polygons[polygonnum].push_back(pm);

		}

	}
	//right button down, clear all the targets in current image
	if (button == GLUT_RIGHT_BUTTON && action == GLUT_DOWN){
		if (mode == 1 && pflag == 1){
			pflag = 0;

			polygonnum++;
			glutPostRedisplay();
		}
	}

}

// mouse movement response function
void PassiveMouseMove(GLint xMouse, GLint yMouse){
	if (rflag == 1 || pflag == 1){
		tempX = xMouse;
		tempY = winHeight - yMouse;
		glutPostRedisplay();
	}

}

// keyboard click response function
void keyboardPress(unsigned char key, int x, int y){
	if (key == 27)	// press 'esc' to quit
		exit(0);
	else if (key > '0' && key <= '9'){	// change target type
		curtype = (int)key - 48;
	}
	else if (key == 'd' || key == 'D'){	// delete last target
		if(mode == 0 && rflag == 0 && targetnum > 0){
			targets.pop_back();
			targetType.pop_back();
			targetnum--;
		}

		if(mode == 1 && rflag == 0 && polygonnum > 0){
			polygons.pop_back();
			polygonType.pop_back();
			polygonnum--;
		}

		glutPostRedisplay();
	}
	else if (key == 'n' || key == 'N'){	// switch to next image
		imgnum++;
		if(imgnum < imgsLen){
			// reset 
			rflag = 0;
			targetnum = 0;
			curtype = 1;
			targets.resize(0);
			targetType.resize(0);

			texture = LoadTexture(imgpaths[imgnum].c_str());
			if(texture){
				sprintf(winName, "image: %d/%d", imgnum+1, imgsLen);
				glutSetWindowTitle(winName);
			}
			else {
				glutSetWindowTitle("fail to load image, press esc to quit...");
			}
		}
		else {
			glutSetWindowTitle("come to the last image,  press esc to quit...");
			imgnum--;
		}
		glutPostRedisplay();

		imshow("instruction", instruction);
		waitKey(10);
	}
	else if(key == 'p' || key == 'P'){	// switch to last image
		imgnum--;
		if(imgnum >= 0){
			// reset 
			rflag = 0;
			targetnum = 0;
			curtype = 1;
			targets.resize(0);
			targetType.resize(0);

			texture = LoadTexture(imgpaths[imgnum].c_str());
			if(texture){
				sprintf(winName, "image: %d/%d", imgnum+1, imgsLen);
				glutSetWindowTitle(winName);
			}
			else {
				glutSetWindowTitle("no image, press esc to quit...");
			}
		}
		else {
			imgnum = 0;
			glutSetWindowTitle("back to the first image...");
		}
		glutPostRedisplay();
	}
	else if (key == 'c' || key == 'C'){
		mode = (mode + 1) % 2;
		curtype = 1;
	}
	else if (key == 'r' || key == 'R'){
		rflag = 0;
		targetnum = 0;
		vector<RectfNode>(targets).swap(targets);	// delete vector
		vector<int>(targetType).swap(targetType);

		pflag = 0;
		polygonnum = 0;
		vector<vector<Point2i> >(polygons).swap(polygons);
		vector<int>(polygonType).swap(polygonType);
		glutPostRedisplay();
	}
	else if (key == 's'){
		saveLabel();
	}
}

int main(int argc, char* argv[]){

	glutInit(&argc, argv);

	initial();

	// initialize window info
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(image_width * wscale, image_height * hscale);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("");

	// callback function
	glutReshapeFunc(ChangeSize);
	glutMouseFunc(MousePlot);
	glutPassiveMotionFunc(PassiveMouseMove);
	glutKeyboardFunc(keyboardPress);
	glutDisplayFunc(Display);

	initialTexture();

	glutMainLoop();

	return 0;
}
