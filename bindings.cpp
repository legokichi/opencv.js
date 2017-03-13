// https://github.com/ucisysarch/opencvjs/blob/master/bindings.cpp

#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include <emscripten/bind.h>

using namespace emscripten;
using namespace cv;
using namespace cv::ml;

namespace Utils{

    template<typename T>
    emscripten::val data(const cv::Mat& mat) {
        return emscripten::val(emscripten::memory_view<T>( (mat.total()*mat.elemSize())/sizeof(T), (T*) mat.data));
    }
    emscripten::val matPtrI(const cv::Mat& mat, int i) {
        return emscripten::val(emscripten::memory_view<uint8_t>(mat.step1(0), mat.ptr(i)));
    }
    emscripten::val matPtrII(const cv::Mat& mat, int i, int j) {
        return emscripten::val(emscripten::memory_view<uint8_t>(mat.step1(1), mat.ptr(i,j)));
    }
    emscripten::val  matFromArray(const emscripten::val& object, int type) {
        int w=  object["width"].as<unsigned>();
        int h=  object["height"].as<unsigned>();
        std::string str = object["data"]["buffer"].as<std::string>();

        return emscripten::val(cv::Mat(h, w, type, (void*)str.data(), 0));
    }
    cv::Mat* createMat(Size size, int type, intptr_t data, size_t step) {
        return new cv::Mat(size, type, reinterpret_cast<void*>(data), step);
    }
    cv::Mat* createMat2(const std::vector<unsigned char>& vector) {
        return new cv::Mat(vector, false);
    }
    static std::vector<int> getMatSize(const cv::Mat& mat)
    {
      std::vector<int> size;
      for (int i = 0; i < mat.dims; i++) {
        size.push_back(mat.size[i]);
      }
      return size;
    }
    static Mat eye(int rows, int cols, int type) {
      return cv::Mat::eye(rows, cols, type);
    }
    static Mat eye(Size size, int type) {
      return cv::Mat::eye(size, type);
    }
    void convertTo(const Mat& obj, Mat& m, int rtype, double alpha, double beta) {
        obj.convertTo(m, rtype, alpha, beta);
    }
    Size matSize(const cv::Mat& mat) {
        return  mat.size();
    }
    cv::Mat zeros(int arg0, int arg1, int arg2) {
        return cv::Mat::zeros(arg0, arg1, arg2);
    }
    cv::Mat zeros(cv::Size arg0, int arg1) {
        return cv::Mat::zeros(arg0,arg1);
    }
    cv::Mat zeros(int arg0, const int* arg1, int arg2) {
        return cv::Mat::zeros(arg0,arg1,arg2);
    }
    cv::Mat ones(int arg0, int arg1, int arg2) {
        return cv::Mat::ones(arg0, arg1, arg2);
    }
    cv::Mat ones(int arg0, const int* arg1, int arg2) {
        return cv::Mat::ones(arg0, arg1, arg2);
    }
    cv::Mat ones(cv::Size arg0, int arg1) {
        return cv::Mat::ones(arg0, arg1);
    }

    double matDot(const cv::Mat& obj, const Mat& mat) {
        return  obj.dot(mat);
    }
    Mat matMul(const cv::Mat& obj, const Mat& mat, double scale) {
        return  Mat(obj.mul(mat, scale));
    }
    Mat matT(const cv::Mat& obj) {
        return  Mat(obj.t());
    }
    Mat matInv(const cv::Mat& obj, int type) {
        return  Mat(obj.inv(type));
    }
}

EMSCRIPTEN_BINDINGS(Utils) {

    register_vector<int>("IntVector");
    register_vector<char>("CharVector");
    register_vector<unsigned>("UnsignedVector");
    register_vector<unsigned char>("UCharVector");
    register_vector<std::string>("StrVector");
    register_vector<emscripten::val>("EmvalVector");
    register_vector<float>("FloatVector");
    register_vector<std::vector<int>>("IntVectorVector");
    register_vector<std::vector<Point>>("PointVectorVector");
    register_vector<cv::Point>("PointVector");
    register_vector<cv::Vec4i>("Vec4iVector");
    register_vector<cv::Mat>("MatVector");
    register_vector<cv::KeyPoint>("KeyPointVector");
    register_vector<cv::Rect>("RectVector");
    register_vector<cv::Point2f>("Point2fVector");

    emscripten::class_<cv::TermCriteria>("TermCriteria")
        .constructor<>()
        .constructor<int, int, double>()
        .property("type", &cv::TermCriteria::type)
        .property("maxCount", &cv::TermCriteria::maxCount)
        .property("epsilon", &cv::TermCriteria::epsilon);

    emscripten::class_<cv::Mat>("Mat")
        .constructor<>()
        //.constructor<const Mat&>()
        .constructor<Size, int>()
        .constructor<int, int, int>()
        .constructor(&Utils::createMat, allow_raw_pointers())
        .constructor(&Utils::createMat2, allow_raw_pointers())
        .function("elemSize1", select_overload<size_t()const>(&cv::Mat::elemSize1))
        //.function("assignTo", select_overload<void(Mat&, int)const>(&cv::Mat::assignTo))
        .function("channels", select_overload<int()const>(&cv::Mat::channels))
        .function("convertTo",  select_overload<void(const Mat&, Mat&, int, double, double)>(&Utils::convertTo))
        .function("total", select_overload<size_t()const>(&cv::Mat::total))
        .function("row", select_overload<Mat(int)const>(&cv::Mat::row))
        .class_function("eye",select_overload<Mat(int, int, int)>(&Utils::eye))
        .class_function("eye",select_overload<Mat(Size, int)>(&Utils::eye))
        .function("create", select_overload<void(int, int, int)>(&cv::Mat::create))
        .function("create", select_overload<void(Size, int)>(&cv::Mat::create))
        .function("rowRange", select_overload<Mat(int, int)const>(&cv::Mat::rowRange))
        .function("rowRange", select_overload<Mat(const Range&)const>(&cv::Mat::rowRange))

        .function("copyTo", select_overload<void(OutputArray)const>(&cv::Mat::copyTo))
        .function("copyTo", select_overload<void(OutputArray, InputArray)const>(&cv::Mat::copyTo))
        .function("elemSize", select_overload<size_t()const>(&cv::Mat::elemSize))

        .function("type", select_overload<int()const>(&cv::Mat::type))
        .function("empty", select_overload<bool()const>(&cv::Mat::empty))
        .function("colRange", select_overload<Mat(int, int)const>(&cv::Mat::colRange))
        .function("colRange", select_overload<Mat(const Range&)const>(&cv::Mat::colRange))
        .function("step1", select_overload<size_t(int)const>(&cv::Mat::step1))
        .function("clone", select_overload<Mat()const>(&cv::Mat::clone))
        .class_function("ones",select_overload<Mat(int, int, int)>(&Utils::ones))
        .class_function("ones",select_overload<Mat(Size, int)>(&Utils::ones))
        .class_function("zeros",select_overload<Mat(int, int, int)>(&Utils::zeros))
        .class_function("zeros",select_overload<Mat(Size, int)>(&Utils::zeros))
        .function("depth", select_overload<int()const>(&cv::Mat::depth))
        .function("col", select_overload<Mat(int)const>(&cv::Mat::col))

        .function("dot", select_overload<double(const Mat&, const Mat&)>(&Utils::matDot))
        .function("mul", select_overload<Mat(const Mat&, const Mat&, double)>(&Utils::matMul))
        .function("inv", select_overload<Mat(const Mat&, int)>(&Utils::matInv))
        .function("t", select_overload<Mat(const Mat&)>(&Utils::matT))

        .property("rows", &cv::Mat::rows)
        .property("cols", &cv::Mat::cols)

        .function("data", &Utils::data<unsigned char>)
        .function("data8S", &Utils::data<char>)
        .function("data16u", &Utils::data<unsigned short>)
        .function("data16s", &Utils::data<short>)
        .function("data32s", &Utils::data<int>)
        .function("data32f", &Utils::data<float>)
        .function("data64f", &Utils::data<double>)

        .function("ptr", select_overload<val(const Mat&, int)>(&Utils::matPtrI))
        .function("ptr", select_overload<val(const Mat&, int, int)>(&Utils::matPtrII))

        .function("size" , &Utils::getMatSize)
        .function("get_uchar_at" , select_overload<unsigned char&(int)>(&cv::Mat::at<unsigned char>))
        .function("get_uchar_at", select_overload<unsigned char&(int, int)>(&cv::Mat::at<unsigned char>))
        .function("get_uchar_at", select_overload<unsigned char&(int, int, int)>(&cv::Mat::at<unsigned char>))
        .function("get_ushort_at", select_overload<unsigned short&(int)>(&cv::Mat::at<unsigned short>))
        .function("get_ushort_at", select_overload<unsigned short&(int, int)>(&cv::Mat::at<unsigned short>))
        .function("get_ushort_at", select_overload<unsigned short&(int, int, int)>(&cv::Mat::at<unsigned short>))
        .function("get_int_at" , select_overload<int&(int)>(&cv::Mat::at<int>) )
        .function("get_int_at", select_overload<int&(int, int)>(&cv::Mat::at<int>) )
        .function("get_int_at", select_overload<int&(int, int, int)>(&cv::Mat::at<int>) )
        .function("get_double_at", select_overload<double&(int, int, int)>(&cv::Mat::at<double>))
        .function("get_double_at", select_overload<double&(int)>(&cv::Mat::at<double>))
        .function("get_double_at", select_overload<double&(int, int)>(&cv::Mat::at<double>))
        .function("get_float_at", select_overload<float&(int)>(&cv::Mat::at<float>))
        .function("get_float_at", select_overload<float&(int, int)>(&cv::Mat::at<float>))
        .function("get_float_at", select_overload<float&(int, int, int)>(&cv::Mat::at<float>))
        .function( "getROI_Rect", select_overload<Mat(const Rect&)const>(&cv::Mat::operator()));

    emscripten::class_<cv::Vec<int,4>>("Vec4i")
        .constructor<>()
        .constructor<int, int, int, int>();

    emscripten::class_<cv::RNG> ("RNG");

    value_array<Size>("Size")
        .element(&Size::height)
        .element(&Size::width);


    value_array<Point>("Point")
        .element(&Point::x)
        .element(&Point::y);

    value_array<Point2f>("Point2f")
        .element(&Point2f::x)
        .element(&Point2f::y);

    emscripten::class_<cv::Rect_<int>> ("Rect")
        .constructor<>()
        .constructor<const cv::Point_<int>&, const cv::Size_<int>&>()
        .constructor<int, int, int, int>()
        .constructor<const cv::Rect_<int>&>()
        .property("x", &cv::Rect_<int>::x)
        .property("y", &cv::Rect_<int>::y)
        .property("width", &cv::Rect_<int>::width)
        .property("height", &cv::Rect_<int>::height);

    emscripten::class_<cv::Scalar_<double>> ("Scalar")
        .constructor<>()
        .constructor<double>()
        .constructor<double, double>()
        .constructor<double, double, double>()
        .constructor<double, double, double, double>()
        .class_function("all", &cv::Scalar_<double>::all)
        .function("isReal", select_overload<bool()const>(&cv::Scalar_<double>::isReal));

    function("matFromArray", &Utils::matFromArray);

    constant("CV_8UC1", CV_8UC1) ;
    constant("CV_8UC2", CV_8UC2) ;
    constant("CV_8UC3", CV_8UC3) ;
    constant("CV_8UC4", CV_8UC4) ;

    constant("CV_8SC1", CV_8SC1) ;
    constant("CV_8SC2", CV_8SC2) ;
    constant("CV_8SC3", CV_8SC3) ;
    constant("CV_8SC4", CV_8SC4) ;

    constant("CV_16UC1", CV_16UC1) ;
    constant("CV_16UC2", CV_16UC2) ;
    constant("CV_16UC3", CV_16UC3) ;
    constant("CV_16UC4", CV_16UC4) ;

    constant("CV_16SC1", CV_16SC1) ;
    constant("CV_16SC2", CV_16SC2) ;
    constant("CV_16SC3", CV_16SC3) ;
    constant("CV_16SC4", CV_16SC4) ;

    constant("CV_32SC1", CV_32SC1) ;
    constant("CV_32SC2", CV_32SC2) ;
    constant("CV_32SC3", CV_32SC3) ;
    constant("CV_32SC4", CV_32SC4) ;

    constant("CV_32FC1", CV_32FC1) ;
    constant("CV_32FC2", CV_32FC2) ;
    constant("CV_32FC3", CV_32FC3) ;
    constant("CV_32FC4", CV_32FC4) ;

    constant("CV_64FC1", CV_64FC1) ;
    constant("CV_64FC2", CV_64FC2) ;
    constant("CV_64FC3", CV_64FC3) ;
    constant("CV_64FC4", CV_64FC4) ;

    constant("CV_8U", CV_8U);
    constant("CV_8S", CV_8S);
    constant("CV_16U", CV_16U);
    constant("CV_16S", CV_16S);
    constant("CV_32S",  CV_32S);
    constant("CV_32F", CV_32F);
    constant("CV_32F", CV_32F);


    constant("BORDER_CONSTANT", +cv::BorderTypes::BORDER_CONSTANT);
    constant("BORDER_REPLICATE", +cv::BorderTypes::BORDER_REPLICATE);
    constant("BORDER_REFLECT", +cv::BorderTypes::BORDER_REFLECT);
    constant("BORDER_WRAP", +cv::BorderTypes::BORDER_WRAP);
    constant("BORDER_REFLECT_101", +cv::BorderTypes::BORDER_REFLECT_101);
    constant("BORDER_TRANSPARENT", +cv::BorderTypes::BORDER_TRANSPARENT);
    constant("BORDER_REFLECT101", +cv::BorderTypes::BORDER_REFLECT101);
    constant("BORDER_DEFAULT", +cv::BorderTypes::BORDER_DEFAULT);
    constant("BORDER_ISOLATED", +cv::BorderTypes::BORDER_ISOLATED);

    constant("NORM_INF", +cv::NormTypes::NORM_INF);
    constant("NORM_L1", +cv::NormTypes::NORM_L1);
    constant("NORM_L2", +cv::NormTypes::NORM_L2);
    constant("NORM_L2SQR", +cv::NormTypes::NORM_L2SQR);
    constant("NORM_HAMMING", +cv::NormTypes::NORM_HAMMING);
    constant("NORM_HAMMING2", +cv::NormTypes::NORM_HAMMING2);
    constant("NORM_TYPE_MASK", +cv::NormTypes::NORM_TYPE_MASK);
    constant("NORM_RELATIVE", +cv::NormTypes::NORM_RELATIVE);
    constant("NORM_MINMAX", +cv::NormTypes::NORM_MINMAX);

}
namespace Wrappers {
    void Canny_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4, int arg5, bool arg6) {
        return cv::Canny(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    void cvtColor_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4) {
        return cv::cvtColor(arg1, arg2, arg3, arg4);
    }

    void integral_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::integral(arg1, arg2, arg3);
    }

    void integral_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, int arg4, int arg5) {
        return cv::integral(arg1, arg2, arg3, arg4, arg5);
    }

    void integral_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, int arg5, int arg6) {
        return cv::integral(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void resize_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Size arg3, double arg4, double arg5, int arg6) {
        return cv::resize(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void sqrt_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::sqrt(arg1, arg2);
    }

    bool CascadeClassifier_load_wrapper(cv::CascadeClassifier& arg0 , const std::string& arg1) {
        return arg0.load(arg1);
    }

    bool CascadeClassifier_convert_wrapper(cv::CascadeClassifier& arg0 , const std::string& arg1, const std::string& arg2) {
        return arg0.convert(arg1, arg2);
    }

    bool CascadeClassifier_read_wrapper(cv::CascadeClassifier& arg0 , const FileNode& arg1) {
        return arg0.read(arg1);
    }

    void CascadeClassifier_detectMultiScale_wrapper(cv::CascadeClassifier& arg0 , const cv::Mat& arg1, std::vector<Rect>& arg2, std::vector<int>& arg3, double arg4, int arg5, int arg6, Size arg7, Size arg8) {
        return arg0.detectMultiScale(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    void CascadeClassifier_detectMultiScale_wrapper(cv::CascadeClassifier& arg0 , const cv::Mat& arg1, std::vector<Rect>& arg2, std::vector<int>& arg3, std::vector<double>& arg4, double arg5, int arg6, int arg7, Size arg8, Size arg9, bool arg10) {
        return arg0.detectMultiScale(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }

    void CascadeClassifier_detectMultiScale_wrapper(cv::CascadeClassifier& arg0 , const cv::Mat& arg1, std::vector<Rect>& arg2, double arg3, int arg4, int arg5, Size arg6, Size arg7) {
        return arg0.detectMultiScale(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

}

EMSCRIPTEN_BINDINGS(testBinding) {
    function("Canny", select_overload<void(const cv::Mat&, cv::Mat&, double, double, int, bool)>(&Wrappers::Canny_wrapper));

    function("cvtColor", select_overload<void(const cv::Mat&, cv::Mat&, int, int)>(&Wrappers::cvtColor_wrapper));

    function("integral", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::integral_wrapper));

    function("integral2", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, int, int)>(&Wrappers::integral_wrapper));

    function("integral3", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&, int, int)>(&Wrappers::integral_wrapper));

    function("resize", select_overload<void(const cv::Mat&, cv::Mat&, Size, double, double, int)>(&Wrappers::resize_wrapper));

    function("sqrt", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::sqrt_wrapper));


    emscripten::class_<cv::BaseCascadeClassifier ,base<Algorithm>>("BaseCascadeClassifier");

    emscripten::class_<cv::CascadeClassifier >("CascadeClassifier")
        .function("load", select_overload<bool(cv::CascadeClassifier&,const std::string&)>(&Wrappers::CascadeClassifier_load_wrapper))
        .function("getFeatureType", select_overload<int()const>(&cv::CascadeClassifier::getFeatureType))
        .class_function("convert", select_overload<bool(cv::CascadeClassifier&,const std::string&,const std::string&)>(&Wrappers::CascadeClassifier_convert_wrapper))
        .function("read", select_overload<bool(cv::CascadeClassifier&,const FileNode&)>(&Wrappers::CascadeClassifier_read_wrapper))
        .function("detectMultiScale2", select_overload<void(cv::CascadeClassifier&,const cv::Mat&,std::vector<Rect>&,std::vector<int>&,double,int,int,Size,Size)>(&Wrappers::CascadeClassifier_detectMultiScale_wrapper))
        .function("isOldFormatCascade", select_overload<bool()const>(&cv::CascadeClassifier::isOldFormatCascade))
        .constructor<>()
        .constructor<const String&>()
        .function("detectMultiScale3", select_overload<void(cv::CascadeClassifier&,const cv::Mat&,std::vector<Rect>&,std::vector<int>&,std::vector<double>&,double,int,int,Size,Size,bool)>(&Wrappers::CascadeClassifier_detectMultiScale_wrapper))
        .function("getOriginalWindowSize", select_overload<Size()const>(&cv::CascadeClassifier::getOriginalWindowSize))
        .function("empty", select_overload<bool()const>(&cv::CascadeClassifier::empty))
        .function("detectMultiScale", select_overload<void(cv::CascadeClassifier&,const cv::Mat&,std::vector<Rect>&,double,int,int,Size,Size)>(&Wrappers::CascadeClassifier_detectMultiScale_wrapper));


    emscripten::enum_<ColorConversionCodes>("ColorConversionCodes")
        .value("COLOR_BGR2BGRA", ColorConversionCodes::COLOR_BGR2BGRA)
        .value("COLOR_RGB2RGBA", ColorConversionCodes::COLOR_RGB2RGBA)
        .value("COLOR_BGRA2BGR", ColorConversionCodes::COLOR_BGRA2BGR)
        .value("COLOR_RGBA2RGB", ColorConversionCodes::COLOR_RGBA2RGB)
        .value("COLOR_BGR2RGBA", ColorConversionCodes::COLOR_BGR2RGBA)
        .value("COLOR_RGB2BGRA", ColorConversionCodes::COLOR_RGB2BGRA)
        .value("COLOR_RGBA2BGR", ColorConversionCodes::COLOR_RGBA2BGR)
        .value("COLOR_BGRA2RGB", ColorConversionCodes::COLOR_BGRA2RGB)
        .value("COLOR_BGR2RGB", ColorConversionCodes::COLOR_BGR2RGB)
        .value("COLOR_RGB2BGR", ColorConversionCodes::COLOR_RGB2BGR)
        .value("COLOR_BGRA2RGBA", ColorConversionCodes::COLOR_BGRA2RGBA)
        .value("COLOR_RGBA2BGRA", ColorConversionCodes::COLOR_RGBA2BGRA)
        .value("COLOR_BGR2GRAY", ColorConversionCodes::COLOR_BGR2GRAY)
        .value("COLOR_RGB2GRAY", ColorConversionCodes::COLOR_RGB2GRAY)
        .value("COLOR_GRAY2BGR", ColorConversionCodes::COLOR_GRAY2BGR)
        .value("COLOR_GRAY2RGB", ColorConversionCodes::COLOR_GRAY2RGB)
        .value("COLOR_GRAY2BGRA", ColorConversionCodes::COLOR_GRAY2BGRA)
        .value("COLOR_GRAY2RGBA", ColorConversionCodes::COLOR_GRAY2RGBA)
        .value("COLOR_BGRA2GRAY", ColorConversionCodes::COLOR_BGRA2GRAY)
        .value("COLOR_RGBA2GRAY", ColorConversionCodes::COLOR_RGBA2GRAY)
        .value("COLOR_BGR2BGR565", ColorConversionCodes::COLOR_BGR2BGR565)
        .value("COLOR_RGB2BGR565", ColorConversionCodes::COLOR_RGB2BGR565)
        .value("COLOR_BGR5652BGR", ColorConversionCodes::COLOR_BGR5652BGR)
        .value("COLOR_BGR5652RGB", ColorConversionCodes::COLOR_BGR5652RGB)
        .value("COLOR_BGRA2BGR565", ColorConversionCodes::COLOR_BGRA2BGR565)
        .value("COLOR_RGBA2BGR565", ColorConversionCodes::COLOR_RGBA2BGR565)
        .value("COLOR_BGR5652BGRA", ColorConversionCodes::COLOR_BGR5652BGRA)
        .value("COLOR_BGR5652RGBA", ColorConversionCodes::COLOR_BGR5652RGBA)
        .value("COLOR_GRAY2BGR565", ColorConversionCodes::COLOR_GRAY2BGR565)
        .value("COLOR_BGR5652GRAY", ColorConversionCodes::COLOR_BGR5652GRAY)
        .value("COLOR_BGR2BGR555", ColorConversionCodes::COLOR_BGR2BGR555)
        .value("COLOR_RGB2BGR555", ColorConversionCodes::COLOR_RGB2BGR555)
        .value("COLOR_BGR5552BGR", ColorConversionCodes::COLOR_BGR5552BGR)
        .value("COLOR_BGR5552RGB", ColorConversionCodes::COLOR_BGR5552RGB)
        .value("COLOR_BGRA2BGR555", ColorConversionCodes::COLOR_BGRA2BGR555)
        .value("COLOR_RGBA2BGR555", ColorConversionCodes::COLOR_RGBA2BGR555)
        .value("COLOR_BGR5552BGRA", ColorConversionCodes::COLOR_BGR5552BGRA)
        .value("COLOR_BGR5552RGBA", ColorConversionCodes::COLOR_BGR5552RGBA)
        .value("COLOR_GRAY2BGR555", ColorConversionCodes::COLOR_GRAY2BGR555)
        .value("COLOR_BGR5552GRAY", ColorConversionCodes::COLOR_BGR5552GRAY)
        .value("COLOR_BGR2XYZ", ColorConversionCodes::COLOR_BGR2XYZ)
        .value("COLOR_RGB2XYZ", ColorConversionCodes::COLOR_RGB2XYZ)
        .value("COLOR_XYZ2BGR", ColorConversionCodes::COLOR_XYZ2BGR)
        .value("COLOR_XYZ2RGB", ColorConversionCodes::COLOR_XYZ2RGB)
        .value("COLOR_BGR2YCrCb", ColorConversionCodes::COLOR_BGR2YCrCb)
        .value("COLOR_RGB2YCrCb", ColorConversionCodes::COLOR_RGB2YCrCb)
        .value("COLOR_YCrCb2BGR", ColorConversionCodes::COLOR_YCrCb2BGR)
        .value("COLOR_YCrCb2RGB", ColorConversionCodes::COLOR_YCrCb2RGB)
        .value("COLOR_BGR2HSV", ColorConversionCodes::COLOR_BGR2HSV)
        .value("COLOR_RGB2HSV", ColorConversionCodes::COLOR_RGB2HSV)
        .value("COLOR_BGR2Lab", ColorConversionCodes::COLOR_BGR2Lab)
        .value("COLOR_RGB2Lab", ColorConversionCodes::COLOR_RGB2Lab)
        .value("COLOR_BGR2Luv", ColorConversionCodes::COLOR_BGR2Luv)
        .value("COLOR_RGB2Luv", ColorConversionCodes::COLOR_RGB2Luv)
        .value("COLOR_BGR2HLS", ColorConversionCodes::COLOR_BGR2HLS)
        .value("COLOR_RGB2HLS", ColorConversionCodes::COLOR_RGB2HLS)
        .value("COLOR_HSV2BGR", ColorConversionCodes::COLOR_HSV2BGR)
        .value("COLOR_HSV2RGB", ColorConversionCodes::COLOR_HSV2RGB)
        .value("COLOR_Lab2BGR", ColorConversionCodes::COLOR_Lab2BGR)
        .value("COLOR_Lab2RGB", ColorConversionCodes::COLOR_Lab2RGB)
        .value("COLOR_Luv2BGR", ColorConversionCodes::COLOR_Luv2BGR)
        .value("COLOR_Luv2RGB", ColorConversionCodes::COLOR_Luv2RGB)
        .value("COLOR_HLS2BGR", ColorConversionCodes::COLOR_HLS2BGR)
        .value("COLOR_HLS2RGB", ColorConversionCodes::COLOR_HLS2RGB)
        .value("COLOR_BGR2HSV_FULL", ColorConversionCodes::COLOR_BGR2HSV_FULL)
        .value("COLOR_RGB2HSV_FULL", ColorConversionCodes::COLOR_RGB2HSV_FULL)
        .value("COLOR_BGR2HLS_FULL", ColorConversionCodes::COLOR_BGR2HLS_FULL)
        .value("COLOR_RGB2HLS_FULL", ColorConversionCodes::COLOR_RGB2HLS_FULL)
        .value("COLOR_HSV2BGR_FULL", ColorConversionCodes::COLOR_HSV2BGR_FULL)
        .value("COLOR_HSV2RGB_FULL", ColorConversionCodes::COLOR_HSV2RGB_FULL)
        .value("COLOR_HLS2BGR_FULL", ColorConversionCodes::COLOR_HLS2BGR_FULL)
        .value("COLOR_HLS2RGB_FULL", ColorConversionCodes::COLOR_HLS2RGB_FULL)
        .value("COLOR_LBGR2Lab", ColorConversionCodes::COLOR_LBGR2Lab)
        .value("COLOR_LRGB2Lab", ColorConversionCodes::COLOR_LRGB2Lab)
        .value("COLOR_LBGR2Luv", ColorConversionCodes::COLOR_LBGR2Luv)
        .value("COLOR_LRGB2Luv", ColorConversionCodes::COLOR_LRGB2Luv)
        .value("COLOR_Lab2LBGR", ColorConversionCodes::COLOR_Lab2LBGR)
        .value("COLOR_Lab2LRGB", ColorConversionCodes::COLOR_Lab2LRGB)
        .value("COLOR_Luv2LBGR", ColorConversionCodes::COLOR_Luv2LBGR)
        .value("COLOR_Luv2LRGB", ColorConversionCodes::COLOR_Luv2LRGB)
        .value("COLOR_BGR2YUV", ColorConversionCodes::COLOR_BGR2YUV)
        .value("COLOR_RGB2YUV", ColorConversionCodes::COLOR_RGB2YUV)
        .value("COLOR_YUV2BGR", ColorConversionCodes::COLOR_YUV2BGR)
        .value("COLOR_YUV2RGB", ColorConversionCodes::COLOR_YUV2RGB)
        .value("COLOR_YUV2RGB_NV12", ColorConversionCodes::COLOR_YUV2RGB_NV12)
        .value("COLOR_YUV2BGR_NV12", ColorConversionCodes::COLOR_YUV2BGR_NV12)
        .value("COLOR_YUV2RGB_NV21", ColorConversionCodes::COLOR_YUV2RGB_NV21)
        .value("COLOR_YUV2BGR_NV21", ColorConversionCodes::COLOR_YUV2BGR_NV21)
        .value("COLOR_YUV420sp2RGB", ColorConversionCodes::COLOR_YUV420sp2RGB)
        .value("COLOR_YUV420sp2BGR", ColorConversionCodes::COLOR_YUV420sp2BGR)
        .value("COLOR_YUV2RGBA_NV12", ColorConversionCodes::COLOR_YUV2RGBA_NV12)
        .value("COLOR_YUV2BGRA_NV12", ColorConversionCodes::COLOR_YUV2BGRA_NV12)
        .value("COLOR_YUV2RGBA_NV21", ColorConversionCodes::COLOR_YUV2RGBA_NV21)
        .value("COLOR_YUV2BGRA_NV21", ColorConversionCodes::COLOR_YUV2BGRA_NV21)
        .value("COLOR_YUV420sp2RGBA", ColorConversionCodes::COLOR_YUV420sp2RGBA)
        .value("COLOR_YUV420sp2BGRA", ColorConversionCodes::COLOR_YUV420sp2BGRA)
        .value("COLOR_YUV2RGB_YV12", ColorConversionCodes::COLOR_YUV2RGB_YV12)
        .value("COLOR_YUV2BGR_YV12", ColorConversionCodes::COLOR_YUV2BGR_YV12)
        .value("COLOR_YUV2RGB_IYUV", ColorConversionCodes::COLOR_YUV2RGB_IYUV)
        .value("COLOR_YUV2BGR_IYUV", ColorConversionCodes::COLOR_YUV2BGR_IYUV)
        .value("COLOR_YUV2RGB_I420", ColorConversionCodes::COLOR_YUV2RGB_I420)
        .value("COLOR_YUV2BGR_I420", ColorConversionCodes::COLOR_YUV2BGR_I420)
        .value("COLOR_YUV420p2RGB", ColorConversionCodes::COLOR_YUV420p2RGB)
        .value("COLOR_YUV420p2BGR", ColorConversionCodes::COLOR_YUV420p2BGR)
        .value("COLOR_YUV2RGBA_YV12", ColorConversionCodes::COLOR_YUV2RGBA_YV12)
        .value("COLOR_YUV2BGRA_YV12", ColorConversionCodes::COLOR_YUV2BGRA_YV12)
        .value("COLOR_YUV2RGBA_IYUV", ColorConversionCodes::COLOR_YUV2RGBA_IYUV)
        .value("COLOR_YUV2BGRA_IYUV", ColorConversionCodes::COLOR_YUV2BGRA_IYUV)
        .value("COLOR_YUV2RGBA_I420", ColorConversionCodes::COLOR_YUV2RGBA_I420)
        .value("COLOR_YUV2BGRA_I420", ColorConversionCodes::COLOR_YUV2BGRA_I420)
        .value("COLOR_YUV420p2RGBA", ColorConversionCodes::COLOR_YUV420p2RGBA)
        .value("COLOR_YUV420p2BGRA", ColorConversionCodes::COLOR_YUV420p2BGRA)
        .value("COLOR_YUV2GRAY_420", ColorConversionCodes::COLOR_YUV2GRAY_420)
        .value("COLOR_YUV2GRAY_NV21", ColorConversionCodes::COLOR_YUV2GRAY_NV21)
        .value("COLOR_YUV2GRAY_NV12", ColorConversionCodes::COLOR_YUV2GRAY_NV12)
        .value("COLOR_YUV2GRAY_YV12", ColorConversionCodes::COLOR_YUV2GRAY_YV12)
        .value("COLOR_YUV2GRAY_IYUV", ColorConversionCodes::COLOR_YUV2GRAY_IYUV)
        .value("COLOR_YUV2GRAY_I420", ColorConversionCodes::COLOR_YUV2GRAY_I420)
        .value("COLOR_YUV420sp2GRAY", ColorConversionCodes::COLOR_YUV420sp2GRAY)
        .value("COLOR_YUV420p2GRAY", ColorConversionCodes::COLOR_YUV420p2GRAY)
        .value("COLOR_YUV2RGB_UYVY", ColorConversionCodes::COLOR_YUV2RGB_UYVY)
        .value("COLOR_YUV2BGR_UYVY", ColorConversionCodes::COLOR_YUV2BGR_UYVY)
        .value("COLOR_YUV2RGB_Y422", ColorConversionCodes::COLOR_YUV2RGB_Y422)
        .value("COLOR_YUV2BGR_Y422", ColorConversionCodes::COLOR_YUV2BGR_Y422)
        .value("COLOR_YUV2RGB_UYNV", ColorConversionCodes::COLOR_YUV2RGB_UYNV)
        .value("COLOR_YUV2BGR_UYNV", ColorConversionCodes::COLOR_YUV2BGR_UYNV)
        .value("COLOR_YUV2RGBA_UYVY", ColorConversionCodes::COLOR_YUV2RGBA_UYVY)
        .value("COLOR_YUV2BGRA_UYVY", ColorConversionCodes::COLOR_YUV2BGRA_UYVY)
        .value("COLOR_YUV2RGBA_Y422", ColorConversionCodes::COLOR_YUV2RGBA_Y422)
        .value("COLOR_YUV2BGRA_Y422", ColorConversionCodes::COLOR_YUV2BGRA_Y422)
        .value("COLOR_YUV2RGBA_UYNV", ColorConversionCodes::COLOR_YUV2RGBA_UYNV)
        .value("COLOR_YUV2BGRA_UYNV", ColorConversionCodes::COLOR_YUV2BGRA_UYNV)
        .value("COLOR_YUV2RGB_YUY2", ColorConversionCodes::COLOR_YUV2RGB_YUY2)
        .value("COLOR_YUV2BGR_YUY2", ColorConversionCodes::COLOR_YUV2BGR_YUY2)
        .value("COLOR_YUV2RGB_YVYU", ColorConversionCodes::COLOR_YUV2RGB_YVYU)
        .value("COLOR_YUV2BGR_YVYU", ColorConversionCodes::COLOR_YUV2BGR_YVYU)
        .value("COLOR_YUV2RGB_YUYV", ColorConversionCodes::COLOR_YUV2RGB_YUYV)
        .value("COLOR_YUV2BGR_YUYV", ColorConversionCodes::COLOR_YUV2BGR_YUYV)
        .value("COLOR_YUV2RGB_YUNV", ColorConversionCodes::COLOR_YUV2RGB_YUNV)
        .value("COLOR_YUV2BGR_YUNV", ColorConversionCodes::COLOR_YUV2BGR_YUNV)
        .value("COLOR_YUV2RGBA_YUY2", ColorConversionCodes::COLOR_YUV2RGBA_YUY2)
        .value("COLOR_YUV2BGRA_YUY2", ColorConversionCodes::COLOR_YUV2BGRA_YUY2)
        .value("COLOR_YUV2RGBA_YVYU", ColorConversionCodes::COLOR_YUV2RGBA_YVYU)
        .value("COLOR_YUV2BGRA_YVYU", ColorConversionCodes::COLOR_YUV2BGRA_YVYU)
        .value("COLOR_YUV2RGBA_YUYV", ColorConversionCodes::COLOR_YUV2RGBA_YUYV)
        .value("COLOR_YUV2BGRA_YUYV", ColorConversionCodes::COLOR_YUV2BGRA_YUYV)
        .value("COLOR_YUV2RGBA_YUNV", ColorConversionCodes::COLOR_YUV2RGBA_YUNV)
        .value("COLOR_YUV2BGRA_YUNV", ColorConversionCodes::COLOR_YUV2BGRA_YUNV)
        .value("COLOR_YUV2GRAY_UYVY", ColorConversionCodes::COLOR_YUV2GRAY_UYVY)
        .value("COLOR_YUV2GRAY_YUY2", ColorConversionCodes::COLOR_YUV2GRAY_YUY2)
        .value("COLOR_YUV2GRAY_Y422", ColorConversionCodes::COLOR_YUV2GRAY_Y422)
        .value("COLOR_YUV2GRAY_UYNV", ColorConversionCodes::COLOR_YUV2GRAY_UYNV)
        .value("COLOR_YUV2GRAY_YVYU", ColorConversionCodes::COLOR_YUV2GRAY_YVYU)
        .value("COLOR_YUV2GRAY_YUYV", ColorConversionCodes::COLOR_YUV2GRAY_YUYV)
        .value("COLOR_YUV2GRAY_YUNV", ColorConversionCodes::COLOR_YUV2GRAY_YUNV)
        .value("COLOR_RGBA2mRGBA", ColorConversionCodes::COLOR_RGBA2mRGBA)
        .value("COLOR_mRGBA2RGBA", ColorConversionCodes::COLOR_mRGBA2RGBA)
        .value("COLOR_RGB2YUV_I420", ColorConversionCodes::COLOR_RGB2YUV_I420)
        .value("COLOR_BGR2YUV_I420", ColorConversionCodes::COLOR_BGR2YUV_I420)
        .value("COLOR_RGB2YUV_IYUV", ColorConversionCodes::COLOR_RGB2YUV_IYUV)
        .value("COLOR_BGR2YUV_IYUV", ColorConversionCodes::COLOR_BGR2YUV_IYUV)
        .value("COLOR_RGBA2YUV_I420", ColorConversionCodes::COLOR_RGBA2YUV_I420)
        .value("COLOR_BGRA2YUV_I420", ColorConversionCodes::COLOR_BGRA2YUV_I420)
        .value("COLOR_RGBA2YUV_IYUV", ColorConversionCodes::COLOR_RGBA2YUV_IYUV)
        .value("COLOR_BGRA2YUV_IYUV", ColorConversionCodes::COLOR_BGRA2YUV_IYUV)
        .value("COLOR_RGB2YUV_YV12", ColorConversionCodes::COLOR_RGB2YUV_YV12)
        .value("COLOR_BGR2YUV_YV12", ColorConversionCodes::COLOR_BGR2YUV_YV12)
        .value("COLOR_RGBA2YUV_YV12", ColorConversionCodes::COLOR_RGBA2YUV_YV12)
        .value("COLOR_BGRA2YUV_YV12", ColorConversionCodes::COLOR_BGRA2YUV_YV12)
        .value("COLOR_BayerBG2BGR", ColorConversionCodes::COLOR_BayerBG2BGR)
        .value("COLOR_BayerGB2BGR", ColorConversionCodes::COLOR_BayerGB2BGR)
        .value("COLOR_BayerRG2BGR", ColorConversionCodes::COLOR_BayerRG2BGR)
        .value("COLOR_BayerGR2BGR", ColorConversionCodes::COLOR_BayerGR2BGR)
        .value("COLOR_BayerBG2RGB", ColorConversionCodes::COLOR_BayerBG2RGB)
        .value("COLOR_BayerGB2RGB", ColorConversionCodes::COLOR_BayerGB2RGB)
        .value("COLOR_BayerRG2RGB", ColorConversionCodes::COLOR_BayerRG2RGB)
        .value("COLOR_BayerGR2RGB", ColorConversionCodes::COLOR_BayerGR2RGB)
        .value("COLOR_BayerBG2GRAY", ColorConversionCodes::COLOR_BayerBG2GRAY)
        .value("COLOR_BayerGB2GRAY", ColorConversionCodes::COLOR_BayerGB2GRAY)
        .value("COLOR_BayerRG2GRAY", ColorConversionCodes::COLOR_BayerRG2GRAY)
        .value("COLOR_BayerGR2GRAY", ColorConversionCodes::COLOR_BayerGR2GRAY)
        .value("COLOR_BayerBG2BGR_VNG", ColorConversionCodes::COLOR_BayerBG2BGR_VNG)
        .value("COLOR_BayerGB2BGR_VNG", ColorConversionCodes::COLOR_BayerGB2BGR_VNG)
        .value("COLOR_BayerRG2BGR_VNG", ColorConversionCodes::COLOR_BayerRG2BGR_VNG)
        .value("COLOR_BayerGR2BGR_VNG", ColorConversionCodes::COLOR_BayerGR2BGR_VNG)
        .value("COLOR_BayerBG2RGB_VNG", ColorConversionCodes::COLOR_BayerBG2RGB_VNG)
        .value("COLOR_BayerGB2RGB_VNG", ColorConversionCodes::COLOR_BayerGB2RGB_VNG)
        .value("COLOR_BayerRG2RGB_VNG", ColorConversionCodes::COLOR_BayerRG2RGB_VNG)
        .value("COLOR_BayerGR2RGB_VNG", ColorConversionCodes::COLOR_BayerGR2RGB_VNG)
        .value("COLOR_BayerBG2BGR_EA", ColorConversionCodes::COLOR_BayerBG2BGR_EA)
        .value("COLOR_BayerGB2BGR_EA", ColorConversionCodes::COLOR_BayerGB2BGR_EA)
        .value("COLOR_BayerRG2BGR_EA", ColorConversionCodes::COLOR_BayerRG2BGR_EA)
        .value("COLOR_BayerGR2BGR_EA", ColorConversionCodes::COLOR_BayerGR2BGR_EA)
        .value("COLOR_BayerBG2RGB_EA", ColorConversionCodes::COLOR_BayerBG2RGB_EA)
        .value("COLOR_BayerGB2RGB_EA", ColorConversionCodes::COLOR_BayerGB2RGB_EA)
        .value("COLOR_BayerRG2RGB_EA", ColorConversionCodes::COLOR_BayerRG2RGB_EA)
        .value("COLOR_BayerGR2RGB_EA", ColorConversionCodes::COLOR_BayerGR2RGB_EA)
        .value("COLOR_COLORCVT_MAX", ColorConversionCodes::COLOR_COLORCVT_MAX);

    emscripten::enum_<ColormapTypes>("ColormapTypes")
        .value("COLORMAP_AUTUMN", ColormapTypes::COLORMAP_AUTUMN)
        .value("COLORMAP_BONE", ColormapTypes::COLORMAP_BONE)
        .value("COLORMAP_JET", ColormapTypes::COLORMAP_JET)
        .value("COLORMAP_WINTER", ColormapTypes::COLORMAP_WINTER)
        .value("COLORMAP_RAINBOW", ColormapTypes::COLORMAP_RAINBOW)
        .value("COLORMAP_OCEAN", ColormapTypes::COLORMAP_OCEAN)
        .value("COLORMAP_SUMMER", ColormapTypes::COLORMAP_SUMMER)
        .value("COLORMAP_SPRING", ColormapTypes::COLORMAP_SPRING)
        .value("COLORMAP_COOL", ColormapTypes::COLORMAP_COOL)
        .value("COLORMAP_HSV", ColormapTypes::COLORMAP_HSV)
        .value("COLORMAP_PINK", ColormapTypes::COLORMAP_PINK)
        .value("COLORMAP_HOT", ColormapTypes::COLORMAP_HOT)
        .value("COLORMAP_PARULA", ColormapTypes::COLORMAP_PARULA);

    emscripten::enum_<ConnectedComponentsTypes>("ConnectedComponentsTypes")
        .value("CC_STAT_LEFT", ConnectedComponentsTypes::CC_STAT_LEFT)
        .value("CC_STAT_TOP", ConnectedComponentsTypes::CC_STAT_TOP)
        .value("CC_STAT_WIDTH", ConnectedComponentsTypes::CC_STAT_WIDTH)
        .value("CC_STAT_HEIGHT", ConnectedComponentsTypes::CC_STAT_HEIGHT)
        .value("CC_STAT_AREA", ConnectedComponentsTypes::CC_STAT_AREA)
        .value("CC_STAT_MAX", ConnectedComponentsTypes::CC_STAT_MAX);

    emscripten::enum_<ContourApproximationModes>("ContourApproximationModes")
        .value("CHAIN_APPROX_NONE", ContourApproximationModes::CHAIN_APPROX_NONE)
        .value("CHAIN_APPROX_SIMPLE", ContourApproximationModes::CHAIN_APPROX_SIMPLE)
        .value("CHAIN_APPROX_TC89_L1", ContourApproximationModes::CHAIN_APPROX_TC89_L1)
        .value("CHAIN_APPROX_TC89_KCOS", ContourApproximationModes::CHAIN_APPROX_TC89_KCOS);

    emscripten::enum_<CovarFlags>("CovarFlags")
        .value("COVAR_SCRAMBLED", CovarFlags::COVAR_SCRAMBLED)
        .value("COVAR_NORMAL", CovarFlags::COVAR_NORMAL)
        .value("COVAR_USE_AVG", CovarFlags::COVAR_USE_AVG)
        .value("COVAR_SCALE", CovarFlags::COVAR_SCALE)
        .value("COVAR_ROWS", CovarFlags::COVAR_ROWS)
        .value("COVAR_COLS", CovarFlags::COVAR_COLS);

    emscripten::enum_<DistanceTransformLabelTypes>("DistanceTransformLabelTypes")
        .value("DIST_LABEL_CCOMP", DistanceTransformLabelTypes::DIST_LABEL_CCOMP)
        .value("DIST_LABEL_PIXEL", DistanceTransformLabelTypes::DIST_LABEL_PIXEL);

    emscripten::enum_<DistanceTransformMasks>("DistanceTransformMasks")
        .value("DIST_MASK_3", DistanceTransformMasks::DIST_MASK_3)
        .value("DIST_MASK_5", DistanceTransformMasks::DIST_MASK_5)
        .value("DIST_MASK_PRECISE", DistanceTransformMasks::DIST_MASK_PRECISE);

    emscripten::enum_<DistanceTypes>("DistanceTypes")
        .value("DIST_USER", DistanceTypes::DIST_USER)
        .value("DIST_L1", DistanceTypes::DIST_L1)
        .value("DIST_L2", DistanceTypes::DIST_L2)
        .value("DIST_C", DistanceTypes::DIST_C)
        .value("DIST_L12", DistanceTypes::DIST_L12)
        .value("DIST_FAIR", DistanceTypes::DIST_FAIR)
        .value("DIST_WELSCH", DistanceTypes::DIST_WELSCH)
        .value("DIST_HUBER", DistanceTypes::DIST_HUBER);

    emscripten::enum_<FloodFillFlags>("FloodFillFlags")
        .value("FLOODFILL_FIXED_RANGE", FloodFillFlags::FLOODFILL_FIXED_RANGE)
        .value("FLOODFILL_MASK_ONLY", FloodFillFlags::FLOODFILL_MASK_ONLY);

    emscripten::enum_<GrabCutClasses>("GrabCutClasses")
        .value("GC_BGD", GrabCutClasses::GC_BGD)
        .value("GC_FGD", GrabCutClasses::GC_FGD)
        .value("GC_PR_BGD", GrabCutClasses::GC_PR_BGD)
        .value("GC_PR_FGD", GrabCutClasses::GC_PR_FGD);

    emscripten::enum_<GrabCutModes>("GrabCutModes")
        .value("GC_INIT_WITH_RECT", GrabCutModes::GC_INIT_WITH_RECT)
        .value("GC_INIT_WITH_MASK", GrabCutModes::GC_INIT_WITH_MASK)
        .value("GC_EVAL", GrabCutModes::GC_EVAL);

    emscripten::enum_<HersheyFonts>("HersheyFonts")
        .value("FONT_HERSHEY_SIMPLEX", HersheyFonts::FONT_HERSHEY_SIMPLEX)
        .value("FONT_HERSHEY_PLAIN", HersheyFonts::FONT_HERSHEY_PLAIN)
        .value("FONT_HERSHEY_DUPLEX", HersheyFonts::FONT_HERSHEY_DUPLEX)
        .value("FONT_HERSHEY_COMPLEX", HersheyFonts::FONT_HERSHEY_COMPLEX)
        .value("FONT_HERSHEY_TRIPLEX", HersheyFonts::FONT_HERSHEY_TRIPLEX)
        .value("FONT_HERSHEY_COMPLEX_SMALL", HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL)
        .value("FONT_HERSHEY_SCRIPT_SIMPLEX", HersheyFonts::FONT_HERSHEY_SCRIPT_SIMPLEX)
        .value("FONT_HERSHEY_SCRIPT_COMPLEX", HersheyFonts::FONT_HERSHEY_SCRIPT_COMPLEX)
        .value("FONT_ITALIC", HersheyFonts::FONT_ITALIC);

    emscripten::enum_<HistCompMethods>("HistCompMethods")
        .value("HISTCMP_CORREL", HistCompMethods::HISTCMP_CORREL)
        .value("HISTCMP_CHISQR", HistCompMethods::HISTCMP_CHISQR)
        .value("HISTCMP_INTERSECT", HistCompMethods::HISTCMP_INTERSECT)
        .value("HISTCMP_BHATTACHARYYA", HistCompMethods::HISTCMP_BHATTACHARYYA)
        .value("HISTCMP_HELLINGER", HistCompMethods::HISTCMP_HELLINGER)
        .value("HISTCMP_CHISQR_ALT", HistCompMethods::HISTCMP_CHISQR_ALT)
        .value("HISTCMP_KL_DIV", HistCompMethods::HISTCMP_KL_DIV);

    emscripten::enum_<HoughModes>("HoughModes")
        .value("HOUGH_STANDARD", HoughModes::HOUGH_STANDARD)
        .value("HOUGH_PROBABILISTIC", HoughModes::HOUGH_PROBABILISTIC)
        .value("HOUGH_MULTI_SCALE", HoughModes::HOUGH_MULTI_SCALE)
        .value("HOUGH_GRADIENT", HoughModes::HOUGH_GRADIENT);

    emscripten::enum_<InterpolationFlags>("InterpolationFlags")
        .value("INTER_NEAREST", InterpolationFlags::INTER_NEAREST)
        .value("INTER_LINEAR", InterpolationFlags::INTER_LINEAR)
        .value("INTER_CUBIC", InterpolationFlags::INTER_CUBIC)
        .value("INTER_AREA", InterpolationFlags::INTER_AREA)
        .value("INTER_LANCZOS4", InterpolationFlags::INTER_LANCZOS4)
        .value("INTER_MAX", InterpolationFlags::INTER_MAX)
        .value("WARP_FILL_OUTLIERS", InterpolationFlags::WARP_FILL_OUTLIERS)
        .value("WARP_INVERSE_MAP", InterpolationFlags::WARP_INVERSE_MAP);

    emscripten::enum_<InterpolationMasks>("InterpolationMasks")
        .value("INTER_BITS", InterpolationMasks::INTER_BITS)
        .value("INTER_BITS2", InterpolationMasks::INTER_BITS2)
        .value("INTER_TAB_SIZE", InterpolationMasks::INTER_TAB_SIZE)
        .value("INTER_TAB_SIZE2", InterpolationMasks::INTER_TAB_SIZE2);

    emscripten::enum_<KmeansFlags>("KmeansFlags")
        .value("KMEANS_RANDOM_CENTERS", KmeansFlags::KMEANS_RANDOM_CENTERS)
        .value("KMEANS_PP_CENTERS", KmeansFlags::KMEANS_PP_CENTERS)
        .value("KMEANS_USE_INITIAL_LABELS", KmeansFlags::KMEANS_USE_INITIAL_LABELS);

    emscripten::enum_<LineSegmentDetectorModes>("LineSegmentDetectorModes")
        .value("LSD_REFINE_NONE", LineSegmentDetectorModes::LSD_REFINE_NONE)
        .value("LSD_REFINE_STD", LineSegmentDetectorModes::LSD_REFINE_STD)
        .value("LSD_REFINE_ADV", LineSegmentDetectorModes::LSD_REFINE_ADV);

    emscripten::enum_<LineTypes>("LineTypes")
        .value("FILLED", LineTypes::FILLED)
        .value("LINE_4", LineTypes::LINE_4)
        .value("LINE_8", LineTypes::LINE_8)
        .value("LINE_AA", LineTypes::LINE_AA);

    emscripten::enum_<MarkerTypes>("MarkerTypes")
        .value("MARKER_CROSS", MarkerTypes::MARKER_CROSS)
        .value("MARKER_TILTED_CROSS", MarkerTypes::MARKER_TILTED_CROSS)
        .value("MARKER_STAR", MarkerTypes::MARKER_STAR)
        .value("MARKER_DIAMOND", MarkerTypes::MARKER_DIAMOND)
        .value("MARKER_SQUARE", MarkerTypes::MARKER_SQUARE)
        .value("MARKER_TRIANGLE_UP", MarkerTypes::MARKER_TRIANGLE_UP)
        .value("MARKER_TRIANGLE_DOWN", MarkerTypes::MARKER_TRIANGLE_DOWN);

    emscripten::enum_<MorphShapes>("MorphShapes")
        .value("MORPH_RECT", MorphShapes::MORPH_RECT)
        .value("MORPH_CROSS", MorphShapes::MORPH_CROSS)
        .value("MORPH_ELLIPSE", MorphShapes::MORPH_ELLIPSE);

    emscripten::enum_<MorphTypes>("MorphTypes")
        .value("MORPH_ERODE", MorphTypes::MORPH_ERODE)
        .value("MORPH_DILATE", MorphTypes::MORPH_DILATE)
        .value("MORPH_OPEN", MorphTypes::MORPH_OPEN)
        .value("MORPH_CLOSE", MorphTypes::MORPH_CLOSE)
        .value("MORPH_GRADIENT", MorphTypes::MORPH_GRADIENT)
        .value("MORPH_TOPHAT", MorphTypes::MORPH_TOPHAT)
        .value("MORPH_BLACKHAT", MorphTypes::MORPH_BLACKHAT)
        .value("MORPH_HITMISS", MorphTypes::MORPH_HITMISS);

    emscripten::enum_<PCA::Flags>("PCA_Flags")
        .value("DATA_AS_ROW", PCA::Flags::DATA_AS_ROW)
        .value("DATA_AS_COL", PCA::Flags::DATA_AS_COL)
        .value("USE_AVG", PCA::Flags::USE_AVG);

    emscripten::enum_<RectanglesIntersectTypes>("RectanglesIntersectTypes")
        .value("INTERSECT_NONE", RectanglesIntersectTypes::INTERSECT_NONE)
        .value("INTERSECT_PARTIAL", RectanglesIntersectTypes::INTERSECT_PARTIAL)
        .value("INTERSECT_FULL", RectanglesIntersectTypes::INTERSECT_FULL);

    emscripten::enum_<ReduceTypes>("ReduceTypes")
        .value("REDUCE_SUM", ReduceTypes::REDUCE_SUM)
        .value("REDUCE_AVG", ReduceTypes::REDUCE_AVG)
        .value("REDUCE_MAX", ReduceTypes::REDUCE_MAX)
        .value("REDUCE_MIN", ReduceTypes::REDUCE_MIN);

    emscripten::enum_<RetrievalModes>("RetrievalModes")
        .value("RETR_EXTERNAL", RetrievalModes::RETR_EXTERNAL)
        .value("RETR_LIST", RetrievalModes::RETR_LIST)
        .value("RETR_CCOMP", RetrievalModes::RETR_CCOMP)
        .value("RETR_TREE", RetrievalModes::RETR_TREE)
        .value("RETR_FLOODFILL", RetrievalModes::RETR_FLOODFILL);

    emscripten::enum_<SVD::Flags>("SVD_Flags")
        .value("MODIFY_A", SVD::Flags::MODIFY_A)
        .value("NO_UV", SVD::Flags::NO_UV)
        .value("FULL_UV", SVD::Flags::FULL_UV);

    emscripten::enum_<SortFlags>("SortFlags")
        .value("SORT_EVERY_ROW", SortFlags::SORT_EVERY_ROW)
        .value("SORT_EVERY_COLUMN", SortFlags::SORT_EVERY_COLUMN)
        .value("SORT_ASCENDING", SortFlags::SORT_ASCENDING)
        .value("SORT_DESCENDING", SortFlags::SORT_DESCENDING);

    emscripten::enum_<TemplateMatchModes>("TemplateMatchModes")
        .value("TM_SQDIFF", TemplateMatchModes::TM_SQDIFF)
        .value("TM_SQDIFF_NORMED", TemplateMatchModes::TM_SQDIFF_NORMED)
        .value("TM_CCORR", TemplateMatchModes::TM_CCORR)
        .value("TM_CCORR_NORMED", TemplateMatchModes::TM_CCORR_NORMED)
        .value("TM_CCOEFF", TemplateMatchModes::TM_CCOEFF)
        .value("TM_CCOEFF_NORMED", TemplateMatchModes::TM_CCOEFF_NORMED);

    emscripten::enum_<TermCriteria::Type>("TermCriteria_Type")
        .value("COUNT", TermCriteria::Type::COUNT)
        .value("MAX_ITER", TermCriteria::Type::MAX_ITER)
        .value("EPS", TermCriteria::Type::EPS);

    emscripten::enum_<ThresholdTypes>("ThresholdTypes")
        .value("THRESH_BINARY", ThresholdTypes::THRESH_BINARY)
        .value("THRESH_BINARY_INV", ThresholdTypes::THRESH_BINARY_INV)
        .value("THRESH_TRUNC", ThresholdTypes::THRESH_TRUNC)
        .value("THRESH_TOZERO", ThresholdTypes::THRESH_TOZERO)
        .value("THRESH_TOZERO_INV", ThresholdTypes::THRESH_TOZERO_INV)
        .value("THRESH_MASK", ThresholdTypes::THRESH_MASK)
        .value("THRESH_OTSU", ThresholdTypes::THRESH_OTSU)
        .value("THRESH_TRIANGLE", ThresholdTypes::THRESH_TRIANGLE);

    emscripten::enum_<UMatUsageFlags>("UMatUsageFlags")
        .value("USAGE_DEFAULT", UMatUsageFlags::USAGE_DEFAULT)
        .value("USAGE_ALLOCATE_HOST_MEMORY", UMatUsageFlags::USAGE_ALLOCATE_HOST_MEMORY)
        .value("USAGE_ALLOCATE_DEVICE_MEMORY", UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY)
        .value("USAGE_ALLOCATE_SHARED_MEMORY", UMatUsageFlags::USAGE_ALLOCATE_SHARED_MEMORY)
        .value("__UMAT_USAGE_FLAGS_32BIT", UMatUsageFlags::__UMAT_USAGE_FLAGS_32BIT);

    emscripten::enum_<UndistortTypes>("UndistortTypes")
        .value("PROJ_SPHERICAL_ORTHO", UndistortTypes::PROJ_SPHERICAL_ORTHO)
        .value("PROJ_SPHERICAL_EQRECT", UndistortTypes::PROJ_SPHERICAL_EQRECT);

    emscripten::enum_<ml::ANN_MLP::ActivationFunctions>("ml_ANN_MLP_ActivationFunctions")
        .value("IDENTITY", ml::ANN_MLP::ActivationFunctions::IDENTITY)
        .value("SIGMOID_SYM", ml::ANN_MLP::ActivationFunctions::SIGMOID_SYM)
        .value("GAUSSIAN", ml::ANN_MLP::ActivationFunctions::GAUSSIAN);

    emscripten::enum_<ml::ANN_MLP::TrainFlags>("ml_ANN_MLP_TrainFlags")
        .value("UPDATE_WEIGHTS", ml::ANN_MLP::TrainFlags::UPDATE_WEIGHTS)
        .value("NO_INPUT_SCALE", ml::ANN_MLP::TrainFlags::NO_INPUT_SCALE)
        .value("NO_OUTPUT_SCALE", ml::ANN_MLP::TrainFlags::NO_OUTPUT_SCALE);

    emscripten::enum_<ml::ANN_MLP::TrainingMethods>("ml_ANN_MLP_TrainingMethods")
        .value("BACKPROP", ml::ANN_MLP::TrainingMethods::BACKPROP)
        .value("RPROP", ml::ANN_MLP::TrainingMethods::RPROP);

    emscripten::enum_<ml::Boost::Types>("ml_Boost_Types")
        .value("DISCRETE", ml::Boost::Types::DISCRETE)
        .value("REAL", ml::Boost::Types::REAL)
        .value("LOGIT", ml::Boost::Types::LOGIT)
        .value("GENTLE", ml::Boost::Types::GENTLE);

    emscripten::enum_<ml::DTrees::Flags>("ml_DTrees_Flags")
        .value("PREDICT_AUTO", ml::DTrees::Flags::PREDICT_AUTO)
        .value("PREDICT_SUM", ml::DTrees::Flags::PREDICT_SUM)
        .value("PREDICT_MAX_VOTE", ml::DTrees::Flags::PREDICT_MAX_VOTE)
        .value("PREDICT_MASK", ml::DTrees::Flags::PREDICT_MASK);

    emscripten::enum_<ml::EM::Types>("ml_EM_Types")
        .value("COV_MAT_SPHERICAL", ml::EM::Types::COV_MAT_SPHERICAL)
        .value("COV_MAT_DIAGONAL", ml::EM::Types::COV_MAT_DIAGONAL)
        .value("COV_MAT_GENERIC", ml::EM::Types::COV_MAT_GENERIC)
        .value("COV_MAT_DEFAULT", ml::EM::Types::COV_MAT_DEFAULT);

    emscripten::enum_<ml::ErrorTypes>("ml_ErrorTypes")
        .value("TEST_ERROR", ml::ErrorTypes::TEST_ERROR)
        .value("TRAIN_ERROR", ml::ErrorTypes::TRAIN_ERROR);

    emscripten::enum_<ml::KNearest::Types>("ml_KNearest_Types")
        .value("BRUTE_FORCE", ml::KNearest::Types::BRUTE_FORCE)
        .value("KDTREE", ml::KNearest::Types::KDTREE);

    emscripten::enum_<ml::LogisticRegression::Methods>("ml_LogisticRegression_Methods")
        .value("BATCH", ml::LogisticRegression::Methods::BATCH)
        .value("MINI_BATCH", ml::LogisticRegression::Methods::MINI_BATCH);

    emscripten::enum_<ml::LogisticRegression::RegKinds>("ml_LogisticRegression_RegKinds")
        .value("REG_DISABLE", ml::LogisticRegression::RegKinds::REG_DISABLE)
        .value("REG_L1", ml::LogisticRegression::RegKinds::REG_L1)
        .value("REG_L2", ml::LogisticRegression::RegKinds::REG_L2);

    emscripten::enum_<ml::SVM::KernelTypes>("ml_SVM_KernelTypes")
        .value("CUSTOM", ml::SVM::KernelTypes::CUSTOM)
        .value("LINEAR", ml::SVM::KernelTypes::LINEAR)
        .value("POLY", ml::SVM::KernelTypes::POLY)
        .value("RBF", ml::SVM::KernelTypes::RBF)
        .value("SIGMOID", ml::SVM::KernelTypes::SIGMOID)
        .value("CHI2", ml::SVM::KernelTypes::CHI2)
        .value("INTER", ml::SVM::KernelTypes::INTER);

    emscripten::enum_<ml::SVM::ParamTypes>("ml_SVM_ParamTypes")
        .value("C", ml::SVM::ParamTypes::C)
        .value("GAMMA", ml::SVM::ParamTypes::GAMMA)
        .value("P", ml::SVM::ParamTypes::P)
        .value("NU", ml::SVM::ParamTypes::NU)
        .value("COEF", ml::SVM::ParamTypes::COEF)
        .value("DEGREE", ml::SVM::ParamTypes::DEGREE);

    emscripten::enum_<ml::SVM::Types>("ml_SVM_Types")
        .value("C_SVC", ml::SVM::Types::C_SVC)
        .value("NU_SVC", ml::SVM::Types::NU_SVC)
        .value("ONE_CLASS", ml::SVM::Types::ONE_CLASS)
        .value("EPS_SVR", ml::SVM::Types::EPS_SVR)
        .value("NU_SVR", ml::SVM::Types::NU_SVR);

    emscripten::enum_<ml::SampleTypes>("ml_SampleTypes")
        .value("ROW_SAMPLE", ml::SampleTypes::ROW_SAMPLE)
        .value("COL_SAMPLE", ml::SampleTypes::COL_SAMPLE);

    emscripten::enum_<ml::StatModel::Flags>("ml_StatModel_Flags")
        .value("UPDATE_MODEL", ml::StatModel::Flags::UPDATE_MODEL)
        .value("RAW_OUTPUT", ml::StatModel::Flags::RAW_OUTPUT)
        .value("COMPRESSED_INPUT", ml::StatModel::Flags::COMPRESSED_INPUT)
        .value("PREPROCESSED_INPUT", ml::StatModel::Flags::PREPROCESSED_INPUT);

    emscripten::enum_<ml::VariableTypes>("ml_VariableTypes")
        .value("VAR_NUMERICAL", ml::VariableTypes::VAR_NUMERICAL)
        .value("VAR_ORDERED", ml::VariableTypes::VAR_ORDERED)
        .value("VAR_CATEGORICAL", ml::VariableTypes::VAR_CATEGORICAL);

}