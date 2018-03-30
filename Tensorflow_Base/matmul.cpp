#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif

#include <vector>
#include <eigen/Dense>

#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

#include <dirent.h>


//#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/core/core_c.h>
//#include <opencv2/core/mat.hpp>
//#include <opencv2/core/operations.hpp>
//#include <opencv2/core/types_c.h>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/highgui/highgui_c.h>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgproc/types_c.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/highgui/highgui_c.h>
//#include <opencv2/imgproc/types_c.h>


#include "matmul.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"



using namespace cv;
using namespace std;

using namespace tensorflow;

// Build a computation graph that takes a tensor of shape [?, 2] and
// multiplies it by a hard-coded matrix.
GraphDef CreateGraphDef()
{
    Scope root = Scope::NewRootScope();

    auto X = ops::Placeholder(root.WithOpName("x"), DT_FLOAT,
        ops::Placeholder::Shape({ -1, 2 }));
    auto A = ops::Const(root, { { 3.f, 2.f },{ -1.f, 0.f } });

    auto Y = ops::MatMul(root.WithOpName("y"), A, X,
        ops::MatMul::TransposeB(true));

    GraphDef def;
    TF_CHECK_OK(root.ToGraphDef(&def));

    return def;
}

int handleOfflineStuff() {
	const string m_data_path = "../data/UCF-101/folds/";

	DIR *dir, *dir1;
	struct dirent *ent, *ent1;
	if ((dir = opendir(m_data_path.c_str() )) != NULL) {
		cout << "Entered dir: " << m_data_path << endl;
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
				continue;

			if ((dir1 = opendir((m_data_path + ent->d_name).c_str())) != NULL) {
				
				cout << "Inside loop Entered dir: " << m_data_path + ent->d_name << endl;

				while ((ent1 = readdir(dir1)) != NULL) {
					if (strcmp(ent1->d_name, ".") == 0 || strcmp(ent1->d_name, "..") == 0)
						continue;

					



					extractFeaturesFromVideo(m_data_path + ent->d_name + "/" + ent1->d_name);
					//return 0;

					//printf("%s\n", ent1->d_name);
					//cout << m_data_path + ent->d_name + "/" + ent1->d_name << endl;

				}
			}


			printf("%s\n", ent->d_name);
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		return EXIT_FAILURE;
	}


	
}

int extractFeaturesFromVideo(string path) {
	//string videoFileLocation = "1/v_BrushingTeeth_g01_c03.avi";

	// Open the video for this camera
	auto m_video = VideoCapture(path);
	assert(m_video.isOpened());

	cout << "read video" << endl;

	//// Assess the image size
	//m_plane_size.width = (int)m_video.get(CV_CAP_PROP_FRAME_WIDTH);
	//m_plane_size.height = (int)m_video.get(CV_CAP_PROP_FRAME_HEIGHT);
	//assert(m_plane_size.area() > 0);

	// Get the amount of video frames
	m_video.set(CV_CAP_PROP_POS_AVI_RATIO, 1);  // Go to the end of the video; 1 = 100%
	long m_frame_amount = (long)m_video.get(CV_CAP_PROP_POS_FRAMES);
	cout << "Frames: " << m_frame_amount;
	assert(m_frame_amount > 1);

	m_video.set(CV_CAP_PROP_POS_AVI_RATIO, 0);  // Go back to the start

	m_video.release(); //Re-open the file because _video.set(CV_CAP_PROP_POS_AVI_RATIO, 1) may screw it up
	m_video = cv::VideoCapture(path);

	int frameNumber = (int)(m_frame_amount / 2);

	//m_video.set(CV_CAP_PROP_POS_FRAMES, frameNumber);  // Go back to the start

	Mat m_frame;
	m_video >> m_frame;
	assert(!m_frame.empty());

	//imshow()

	cout << "path: " << path << endl;
	cout << "Framenumber: " << frameNumber << endl;

	//cvShowImage("windows", &m_frame);
	/*imshow("windows", m_frame);
	imshow("windows", m_frame);*/

	Mat gray;

	for (int i = 0; i < m_frame_amount - 1; i++) {
		cout << i << endl;
		m_video >> m_frame;
		assert(!m_frame.empty());
		cvtColor(m_frame, gray, CV_BGR2GRAY);
		imshow("windows", gray);
		//imshow("windows", m_frame);

		//cout << m_frame << endl;
		//cin >> path;
		
	}
	//m_video >> m_frame;
	//imshow("windows", m_frame);



	/*for (int i = 0; i < 1000; i++) {
		cout << "Hmm haha " << i << endl;
	}*/

	//cin >> path;
	return 0;
}

int main()
{
	VideoCapture m_video = VideoCapture(0);

	Mat m_frame;

	for (int i = 0; i < 1000; i++) {
		m_video.read(m_frame);// >> m_frame;
		assert(!m_frame.empty());

		imshow("nothing", m_frame);
		for (int j = 0; j < 5; j++)
			cout << "haha" + j;
	}

	string a;

	cin >> a;




	string answer;
	//Mat a;
	cout << "Do you want to do the offline stuff?" << endl;
	cin >> answer;

	if (answer == "y") {
		cout << "read y " << endl;
		handleOfflineStuff();
	}
	else if (answer == "n") {
		cout << "read n" << endl;
	}


	cin >> answer;
	//switch 
    GraphDef graph_def = CreateGraphDef();

	//ops::Variable()

    // Start up the session
    SessionOptions options;
    std::unique_ptr<Session> session(NewSession(options));
    TF_CHECK_OK(session->Create(graph_def));

    // Define some data.  This needs to be converted to an Eigen Tensor to be
    // fed into the placeholder.  Note that this will be broken up into two
    // separate vectors of length 2: [1, 2] and [3, 4], which will separately
    // be multiplied by the matrix.
    std::vector<float> data = { 1, 2, 3, 4 };
    auto mapped_X_ = Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>>
        (&data[0], 2, 2);
    auto eigen_X_ = Eigen::Tensor<float, 2, Eigen::RowMajor>(mapped_X_);

    Tensor X_(DT_FLOAT, TensorShape({ 2, 2 }));
    X_.tensor<float, 2>() = eigen_X_;

    std::vector<Tensor> outputs;
    TF_CHECK_OK(session->Run({ { "x", X_ } }, { "y" }, {}, &outputs));

    // Get the result and print it out
    Tensor Y_ = outputs[0];
    std::cout << Y_.tensor<float, 2>() << std::endl;

    session->Close();
}