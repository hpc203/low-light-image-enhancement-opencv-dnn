#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class PairLIE
{
public:
	PairLIE(string modelpath, float exposure = 0.5);
	Mat detect(Mat srcimg);
private:
	int inpWidth;
	int inpHeight;
	Mat exposure_;
	Net net;
};

PairLIE::PairLIE(string model_path, float exposure)
{
	this->net = readNet(model_path);

	size_t pos = model_path.rfind("_");
	size_t pos_ = model_path.rfind(".");
	int len = pos_ - pos - 1;
	string hxw = model_path.substr(pos + 1, len);

	pos = hxw.rfind("x");
	string h = hxw.substr(0, pos);
	len = hxw.length() - pos;
	string w = hxw.substr(pos + 1, len);
	this->inpHeight = stoi(h);
	this->inpWidth = stoi(w);
	Mat one = (Mat_<float>(1, 1) << exposure);
	this->exposure_ = blobFromImage(one);
}

Mat PairLIE::detect(Mat srcimg)
{
	const int srch = srcimg.rows;
	const int srcw = srcimg.cols;
	Mat blob = blobFromImage(srcimg, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);

	this->net.setInput(blob, "input");
	this->net.setInput(this->exposure_, "exposure");   ////opencv-dnn多输入代码参考https://github.com/opencv/opencv/issues/19304
	vector<Mat> outs;
	net.enableWinograd(false);  ////如果是opencv4.7，那就需要加上这一行
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	float* pdata = (float*)outs[0].data;
	const int out_h = outs[0].size[2];
	const int out_w = outs[0].size[3];
	const int channel_step = out_h * out_w;
	Mat rmat(out_h, out_w, CV_32FC1, pdata);
	Mat gmat(out_h, out_w, CV_32FC1, pdata + channel_step);
	Mat bmat(out_h, out_w, CV_32FC1, pdata + 2 * channel_step);
	rmat *= 255.f;
	gmat *= 255.f;
	bmat *= 255.f;
	///output_image = 等价np.clip(output_image, 0, 255)
	rmat.setTo(0, rmat < 0);
	rmat.setTo(255, rmat > 255);
	gmat.setTo(0, gmat < 0);
	gmat.setTo(255, gmat > 255);
	bmat.setTo(0, bmat < 0);
	bmat.setTo(255, bmat > 255);

	vector<Mat> channel_mats(3);
	channel_mats[0] = bmat;
	channel_mats[1] = gmat;
	channel_mats[2] = rmat;

	Mat dstimg;
	merge(channel_mats, dstimg);
	dstimg.convertTo(dstimg, CV_8UC3);
	resize(dstimg, dstimg, Size(srcw, srch));
	return dstimg;
}

int main()
{
	PairLIE mynet("weights/pairlie_512x512.onnx");

	string imgpath = "testimgs/1.png";
	Mat srcimg = imread(imgpath);
	Mat dstimg = mynet.detect(srcimg);
	
	namedWindow("srcimg", WINDOW_NORMAL);
	imshow("srcimg", srcimg);
	static const string kWinName = "Deep learning use OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, dstimg);
	waitKey(0);
	destroyAllWindows();
}
