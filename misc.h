#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <string>
#include <algorithm>
#include <pcl/common/transforms.h>
#include "vtk_model_sampling.h"
#define NUM_FRAMES 128
#define SCALE_X 2.7
#define SCALE_Y 2.4
#define SCALE_Z 3.0


void nicePrintInfo(cv::Mat file ) {
    cout<< "Value at CC_STAT_LEFT: " << file.at<int>(0,cv::CC_STAT_LEFT) << std::endl;
    cout<< "Value at CC_STAT_TOP: " << file.at<int>(0,cv::CC_STAT_TOP)<< std::endl;
    cout<< "Value at CC_STAT_WIDTH: " << file.at<int>(0,cv::CC_STAT_WIDTH)<< std::endl;
    cout<< "Value at CC_STAT_HEIGHT: " << file.at<int>(0,cv::CC_STAT_HEIGHT)<< std::endl;
}


void convertVector3fToCl(Eigen::Vector3f vector3f, float *res) {
  res[0]=vector3f[0];
  res[1]=vector3f[1];
  res[2]=vector3f[2];
}

std::vector<float> convertVector4fToCl(Eigen::Vector4f vector4f) {
    std::vector<float> result;
    result.push_back(vector4f[0]);
    result.push_back(vector4f[1]);
    result.push_back(vector4f[2]);
    result.push_back(vector4f[3]);
    return result;
}

void convertPointXYZtoCL(pcl::PointXYZ point, float* result) {
    result[0]= point.x;
    result[1]= point.y;
    result[2]= point.z;
}

void convertMatrix3fToCL(Eigen::Matrix3f matrix3f, float* result) {
		for (int i = 0; i<3; i++) {//row
			for (int k = 0; k<3; k++) { // colm
                                result[i*3+k]=matrix3f(i,k);
			}
		}

}

void convertPointCloudToCL(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, float* res,int size) {
    for (int i = 0 ; i <size ; i++) {
        res[i*3]= pointCloud.get()->at(i).x;
        res[i*3+1]= pointCloud.get()->at(i).y;
        res[i*3+2]= pointCloud.get()->at(i).z;

    }
}
