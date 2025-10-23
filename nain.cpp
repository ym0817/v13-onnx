//libraries included via NuGet
#include "inference.h"
#include "utils.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
    std::string image_path = "D:\\AlgoData\\Trains\\try\\yolov13cpp\\models\\dog.jpg";
    //std::string modelPath = "D:\\AlgoData\\Trains\\yolov13cpp\\models\\yolov12n.onnx";
    
    std::string modelPath = "D:\\AlgoData\\Trains\\try\\yolov13cpp\\models\\yolov13n.onnx";
    //std::string modelPath = "D:\\AlgoData\\Trains\\try\\yolov13cpp\\models\\yolov13s.onnx";
    std::string classNamesPath = "D:\\AlgoData\\Trains\\try\\yolov13cpp\\models\\coco.names";


    //std::string modelPath = "D:\\AlgoData\\Trains\\yolov13cpp\\models\\zd_hz.onnx";
    //std::string classNamesPath = "D:\\AlgoData\\Trains\\yolov13cpp\\models\\SN_class.names";


    float confThreshold = 0.4f;
    float iouThreshold = 0.4f;
    float maskThreshold = 0.5f;
    //bool isGPU = true;
    bool isGPU = false;
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    const std::string suffixName = "m";

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }
    std::shared_ptr<YOLOPredictor> predictor(new YOLOPredictor(modelPath, isGPU,
        confThreshold, iouThreshold, maskThreshold));
    std::vector<YoloResult> outputs;
    //std::vector<types::Boxf> detected_boxes;
    cv::Mat bgr_img = cv::imread(image_path);
    outputs = predictor->predict(bgr_img);
    int type_index = outputs[0].classId;
    std::cerr << "outputs  :   " << type_index << "   , " << classNames[type_index] << std::endl;
    utils::visualizeDetection(bgr_img, outputs, classNames);
    //imshow("result", bgr_img);
    //cv::waitKey(0);


    for (int n = 0; n < 80; n++)
    {

        auto start = std::chrono::steady_clock::now();
        outputs = predictor->predict(bgr_img);
        auto end = std::chrono::steady_clock::now();
        //std::chrono::duration<double> spent = end - start;
        double spent_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "---------PerImage cost time:    " << spent_ms << "   ms " << std::endl;
       
    }




    return 0;
}
