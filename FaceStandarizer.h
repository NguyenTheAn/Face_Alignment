#include <opencv2/opencv.hpp>
#include <string>
#include <queue>
#include <math.h>
#include <vector>

cv::Mat meanAxis0(const cv::Mat &src);
cv::Mat elementwiseMinus(const cv::Mat &A,const cv::Mat &B);
cv::Mat varAxis0(const cv::Mat &src);
int MatrixRank(cv::Mat M);


class FaceStandarizer {
    public:
        FaceStandarizer(int crop_type = 2);
        ~FaceStandarizer();
        cv::Mat estimateTrans(std::vector<cv::Point2f> srcLmks);
        cv::Mat alignFace(const cv::Mat &inputImage, std::vector<cv::Point2f> inputLmks);

    private:
        std::vector<cv::Point2f> targetLmksVec;
};