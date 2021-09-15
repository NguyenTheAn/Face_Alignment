#include "FaceStandarizer.h"
#include <fstream>

class Detector{
    public:
        Detector();
        Detector(const std::string model_name, const std::string detect_landmark);
        cv::CascadeClassifier face_cascade;
        cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create();
        void detect(cv::Mat image, std::vector<cv::Rect> &faces, std::vector< std::vector<cv::Point2f> > &landmarks);
};

int main(int argc, char** argv){

    std::string path;
    if  (argc == 1)
    {
        path = "../test/00001.png";
    }
    else if (argc == 2)
    {
        path = argv[1];
    }
    Detector detector("../model/haarcascade_frontalface_alt2.xml", "../model/lbfmodel.yaml");
    FaceStandarizer faceStandarizer(2);

    cv::Mat frame = cv::imread(path.c_str());
    cv::Mat img = frame.clone();

    std::vector<cv::Rect> faces;
    std::vector< std::vector<cv::Point2f> > landmarks;
    detector.detect(frame, faces, landmarks);
    for (auto face : faces){
        cv::rectangle(frame, face, cv::Scalar(0, 255, 0));
    }
    for(auto landmark : landmarks){
        std::vector<cv::Point2f> landmark_5_points;
        landmark_5_points.push_back(cv::Point2f(((landmark[36] + landmark[39])/2).x, ((landmark[36] + landmark[39])/2).y));
        landmark_5_points.push_back(cv::Point2f(((landmark[42] + landmark[45])/2).x, ((landmark[42] + landmark[45])/2).y));
        landmark_5_points.push_back(cv::Point2f(landmark[30].x, landmark[30].y));
        landmark_5_points.push_back(cv::Point2f(landmark[48].x, landmark[48].y));
        landmark_5_points.push_back(cv::Point2f(landmark[54].x, landmark[54].y));


        cv::Mat warpImg = faceStandarizer.alignFace(img, landmark_5_points);
        cv::circle(frame, (landmark[36] + landmark[39])/2, 3, cv::Scalar(0, 0, 255), cv::FILLED);
        cv::circle(frame, (landmark[42] + landmark[45])/2, 3, cv::Scalar(0, 0, 255), cv::FILLED);
        cv::circle(frame, landmark[30], 3, cv::Scalar(0, 0, 255), cv::FILLED);
        cv::circle(frame, landmark[48], 3, cv::Scalar(0, 0, 255), cv::FILLED);
        cv::circle(frame, landmark[54], 3, cv::Scalar(0, 0, 255), cv::FILLED);
        cv::imshow("warpImg", warpImg);
        cv::imwrite("output.jpg", warpImg);
    }
    
    cv::imshow("frame", frame);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // cv::VideoCapture cam("../phuongpt94_happiness.avi");
    // cv::Mat frame;
    // while(cam.read(frame)){
    //     cv::Mat img = frame.clone();
    //     cv::imwrite("unit_test.jpg", img);
    //     std::vector<cv::Rect> faces;
    //     std::vector< std::vector<cv::Point2f> > landmarks;
    //     detector.detect(frame, faces, landmarks);
    //     for (auto face : faces){
    //         cv::rectangle(frame, face, cv::Scalar(0, 255, 0));
    //     }

    //     for(auto landmark : landmarks){
    //         float landmark_5_points[5][2] = {
    //             {((landmark[36] + landmark[39])/2).x, ((landmark[36] + landmark[39])/2).y},
    //             {((landmark[42] + landmark[45])/2).x, ((landmark[42] + landmark[45])/2).y},
    //             {landmark[30].x, landmark[30].y},
    //             {landmark[48].x, landmark[48].y},
    //             {landmark[54].x, landmark[54].y}};

    //         cv::Mat dst(5,2,CV_32FC1, landmark_5_points);

    //         // cv::Mat m = FaceStandarizer::similarTransform(dst, src);
    //         // cv::Rect roi(0, 0, 3, 2);
    //         // cv::Mat M = m(roi);
    //         // cv::Mat warpImg;
    //         // cv::warpAffine(img, warpImg, M, cv::Size(112, 112));

    //         cv::Mat warpImg = FaceStandarizer::alignFace(img, dst);

    //         // auto t1 =cv::getTickCount();
    //         // int pred = classifier.Classify(warpImg, 5);
    //         // std::string out = mapped[pred];
    //         // auto t2 =cv::getTickCount();
    //         // std::cout<<"Forward time: "<<(t2-t1)/cv::getTickFrequency()*1000<<std::endl;
    //         // cv::putText(frame, out, cv::Point(20, 20), cv::FONT_HERSHEY_DUPLEX, 1, CV_RGB(0, 0, 255), 1);

    //         cv::circle(frame, (landmark[36] + landmark[39])/2, 3, cv::Scalar(0, 0, 255), cv::FILLED);
    //         cv::circle(frame, (landmark[42] + landmark[45])/2, 3, cv::Scalar(0, 0, 255), cv::FILLED);
    //         cv::circle(frame, landmark[30], 3, cv::Scalar(0, 0, 255), cv::FILLED);
    //         cv::circle(frame, landmark[48], 3, cv::Scalar(0, 0, 255), cv::FILLED);
    //         cv::circle(frame, landmark[54], 3, cv::Scalar(0, 0, 255), cv::FILLED);
    //         cv::imshow("warpImg", warpImg);
    //     }

    //     cv::imshow("frame", frame);
    //     char c = cv::waitKey(0);
    //     if (c == 'q') {
    //         cv::destroyAllWindows();
    //         break;
    //     }
    // }
    
    return 0;
}


Detector::Detector(const std::string detect_face, const std::string detect_landmark){
    this->face_cascade.load(detect_face);
    facemark->loadModel(detect_landmark);
}

void Detector::detect(cv::Mat image, std::vector<cv::Rect> &faces, std::vector< std::vector<cv::Point2f> > &landmarks){
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    face_cascade.detectMultiScale(gray, faces);
    facemark->fit(image, faces, landmarks);
}