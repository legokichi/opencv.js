// https://github.com/ucisysarch/opencvjs/blob/master/bindings.cpp

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include <emscripten/bind.h>


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
    cv::Mat* createMat(cv::Size size, int type, intptr_t data, size_t step) {
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
    static cv::Mat eye(int rows, int cols, int type) {
      return cv::Mat::eye(rows, cols, type);
    }
    static cv::Mat eye(cv::Size size, int type) {
      return cv::Mat::eye(size, type);
    }
    void convertTo(const cv::Mat& obj, cv::Mat& m, int rtype, double alpha, double beta) {
        obj.convertTo(m, rtype, alpha, beta);
    }
    cv::Size matSize(const cv::Mat& mat) {
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

    double matDot(const cv::Mat& obj, const cv::Mat& mat) {
        return  obj.dot(mat);
    }
    cv::Mat matMul(const cv::Mat& obj, const cv::Mat& mat, double scale) {
        return  cv::Mat(obj.mul(mat, scale));
    }
    cv::Mat matT(const cv::Mat& obj) {
        return  cv::Mat(obj.t());
    }
    cv::Mat matInv(const cv::Mat& obj, int type) {
        return  cv::Mat(obj.inv(type));
    }
}

EMSCRIPTEN_BINDINGS(Utils) {

    emscripten::register_vector<int>("IntVector");
    emscripten::register_vector<char>("CharVector");
    emscripten::register_vector<unsigned>("UnsignedVector");
    emscripten::register_vector<unsigned char>("UCharVector");
    emscripten::register_vector<std::string>("StrVector");
    emscripten::register_vector<emscripten::val>("EmvalVector");
    emscripten::register_vector<float>("FloatVector");
    emscripten::register_vector<std::vector<int>>("IntVectorVector");
    emscripten::register_vector<std::vector<cv::Point>>("PointVectorVector");
    emscripten::register_vector<cv::Point>("PointVector");
    emscripten::register_vector<cv::Vec4i>("Vec4iVector");
    emscripten::register_vector<cv::Mat>("MatVector");
    emscripten::register_vector<cv::KeyPoint>("KeyPointVector");
    emscripten::register_vector<cv::Rect>("RectVector");
    emscripten::register_vector<cv::Point2f>("Point2fVector");

    emscripten::class_<cv::TermCriteria>("TermCriteria")
        .constructor<>()
        .constructor<int, int, double>()
        .property("type", &cv::TermCriteria::type)
        .property("maxCount", &cv::TermCriteria::maxCount)
        .property("epsilon", &cv::TermCriteria::epsilon);

    emscripten::class_<cv::Mat>("Mat")
        .constructor<>()
        //.constructor<const cv::Mat&>()
        .constructor<cv::Size, int>()
        .constructor<int, int, int>()
        .constructor(&Utils::createMat, emscripten::allow_raw_pointers())
        .constructor(&Utils::createMat2, emscripten::allow_raw_pointers())
        .function("elemSize1", emscripten::select_overload<size_t()const>(&cv::Mat::elemSize1))
        //.function("assignTo", emscripten::select_overload<void(Mat&, int)const>(&cv::Mat::assignTo))
        .function("channels", emscripten::select_overload<int()const>(&cv::Mat::channels))
        .function("convertTo",  emscripten::select_overload<void(const cv::Mat&, cv::Mat&, int, double, double)>(&Utils::convertTo))
        .function("total", emscripten::select_overload<size_t()const>(&cv::Mat::total))
        .function("row", emscripten::select_overload<cv::Mat(int)const>(&cv::Mat::row))
        .class_function("eye", emscripten::select_overload<cv::Mat(int, int, int)>(&Utils::eye))
        .class_function("eye", emscripten::select_overload<cv::Mat(cv::Size, int)>(&Utils::eye))
        .function("create", emscripten::select_overload<void(int, int, int)>(&cv::Mat::create))
        .function("create", emscripten::select_overload<void(cv::Size, int)>(&cv::Mat::create))
        .function("rowRange", emscripten::select_overload<cv::Mat(int, int)const>(&cv::Mat::rowRange))
        .function("rowRange", emscripten::select_overload<cv::Mat(const cv::Range&)const>(&cv::Mat::rowRange))

        .function("copyTo", emscripten::select_overload<void(cv::OutputArray)const>(&cv::Mat::copyTo))
        .function("copyTo", emscripten::select_overload<void(cv::OutputArray, cv::InputArray)const>(&cv::Mat::copyTo))
        .function("elemSize", emscripten::select_overload<size_t()const>(&cv::Mat::elemSize))

        .function("type", emscripten::select_overload<int()const>(&cv::Mat::type))
        .function("empty", emscripten::select_overload<bool()const>(&cv::Mat::empty))
        .function("colRange", emscripten::select_overload<cv::Mat(int, int)const>(&cv::Mat::colRange))
        .function("colRange", emscripten::select_overload<cv::Mat(const cv::Range&)const>(&cv::Mat::colRange))
        .function("step1", emscripten::select_overload<size_t(int)const>(&cv::Mat::step1))
        .function("clone", emscripten::select_overload<cv::Mat()const>(&cv::Mat::clone))
        .class_function("ones", emscripten::select_overload<cv::Mat(int, int, int)>(&Utils::ones))
        .class_function("ones", emscripten::select_overload<cv::Mat(cv::Size, int)>(&Utils::ones))
        .class_function("zeros", emscripten::select_overload<cv::Mat(int, int, int)>(&Utils::zeros))
        .class_function("zeros", emscripten::select_overload<cv::Mat(cv::Size, int)>(&Utils::zeros))
        .function("depth", emscripten::select_overload<int()const>(&cv::Mat::depth))
        .function("col", emscripten::select_overload<cv::Mat(int)const>(&cv::Mat::col))

        .function("dot", emscripten::select_overload<double(const cv::Mat&, const cv::Mat&)>(&Utils::matDot))
        .function("mul", emscripten::select_overload<cv::Mat(const cv::Mat&, const cv::Mat&, double)>(&Utils::matMul))
        .function("inv", emscripten::select_overload<cv::Mat(const cv::Mat&, int)>(&Utils::matInv))
        .function("t", emscripten::select_overload<cv::Mat(const cv::Mat&)>(&Utils::matT))

        .property("rows", &cv::Mat::rows)
        .property("cols", &cv::Mat::cols)

        .function("data", &Utils::data<unsigned char>)
        .function("data8S", &Utils::data<char>)
        .function("data16u", &Utils::data<unsigned short>)
        .function("data16s", &Utils::data<short>)
        .function("data32s", &Utils::data<int>)
        .function("data32f", &Utils::data<float>)
        .function("data64f", &Utils::data<double>)

        .function("ptr", emscripten::select_overload<emscripten::val(const cv::Mat&, int)>(&Utils::matPtrI))
        .function("ptr", emscripten::select_overload<emscripten::val(const cv::Mat&, int, int)>(&Utils::matPtrII))

        .function("size" , &Utils::getMatSize)
        .function("get_uchar_at" , emscripten::select_overload<unsigned char&(int)>(&cv::Mat::at<unsigned char>))
        .function("get_uchar_at", emscripten::select_overload<unsigned char&(int, int)>(&cv::Mat::at<unsigned char>))
        .function("get_uchar_at", emscripten::select_overload<unsigned char&(int, int, int)>(&cv::Mat::at<unsigned char>))
        .function("get_ushort_at", emscripten::select_overload<unsigned short&(int)>(&cv::Mat::at<unsigned short>))
        .function("get_ushort_at", emscripten::select_overload<unsigned short&(int, int)>(&cv::Mat::at<unsigned short>))
        .function("get_ushort_at", emscripten::select_overload<unsigned short&(int, int, int)>(&cv::Mat::at<unsigned short>))
        .function("get_int_at" , emscripten::select_overload<int&(int)>(&cv::Mat::at<int>) )
        .function("get_int_at", emscripten::select_overload<int&(int, int)>(&cv::Mat::at<int>) )
        .function("get_int_at", emscripten::select_overload<int&(int, int, int)>(&cv::Mat::at<int>) )
        .function("get_double_at", emscripten::select_overload<double&(int, int, int)>(&cv::Mat::at<double>))
        .function("get_double_at", emscripten::select_overload<double&(int)>(&cv::Mat::at<double>))
        .function("get_double_at", emscripten::select_overload<double&(int, int)>(&cv::Mat::at<double>))
        .function("get_float_at", emscripten::select_overload<float&(int)>(&cv::Mat::at<float>))
        .function("get_float_at", emscripten::select_overload<float&(int, int)>(&cv::Mat::at<float>))
        .function("get_float_at", emscripten::select_overload<float&(int, int, int)>(&cv::Mat::at<float>))
        .function( "getROI_Rect", emscripten::select_overload<cv::Mat(const cv::Rect&)const>(&cv::Mat::operator()));

    emscripten::class_<cv::Vec<int,4>>("Vec4i")
        .constructor<>()
        .constructor<int, int, int, int>();

    emscripten::class_<cv::RNG> ("RNG");

    emscripten::value_array<cv::Size>("Size")
        .element(&cv::Size::height)
        .element(&cv::Size::width);


    emscripten::value_array<cv::Point>("Point")
        .element(&cv::Point::x)
        .element(&cv::Point::y);

    emscripten::value_array<cv::Point2f>("Point2f")
        .element(&cv::Point2f::x)
        .element(&cv::Point2f::y);

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
        .function("isReal", emscripten::select_overload<bool()const>(&cv::Scalar_<double>::isReal));

    emscripten::function("matFromArray", &Utils::matFromArray);

    emscripten::constant("CV_8UC1", CV_8UC1) ;
    emscripten::constant("CV_8UC2", CV_8UC2) ;
    emscripten::constant("CV_8UC3", CV_8UC3) ;
    emscripten::constant("CV_8UC4", CV_8UC4) ;

    emscripten::constant("CV_8SC1", CV_8SC1) ;
    emscripten::constant("CV_8SC2", CV_8SC2) ;
    emscripten::constant("CV_8SC3", CV_8SC3) ;
    emscripten::constant("CV_8SC4", CV_8SC4) ;

    emscripten::constant("CV_16UC1", CV_16UC1) ;
    emscripten::constant("CV_16UC2", CV_16UC2) ;
    emscripten::constant("CV_16UC3", CV_16UC3) ;
    emscripten::constant("CV_16UC4", CV_16UC4) ;

    emscripten::constant("CV_16SC1", CV_16SC1) ;
    emscripten::constant("CV_16SC2", CV_16SC2) ;
    emscripten::constant("CV_16SC3", CV_16SC3) ;
    emscripten::constant("CV_16SC4", CV_16SC4) ;

    emscripten::constant("CV_32SC1", CV_32SC1) ;
    emscripten::constant("CV_32SC2", CV_32SC2) ;
    emscripten::constant("CV_32SC3", CV_32SC3) ;
    emscripten::constant("CV_32SC4", CV_32SC4) ;

    emscripten::constant("CV_32FC1", CV_32FC1) ;
    emscripten::constant("CV_32FC2", CV_32FC2) ;
    emscripten::constant("CV_32FC3", CV_32FC3) ;
    emscripten::constant("CV_32FC4", CV_32FC4) ;

    emscripten::constant("CV_64FC1", CV_64FC1) ;
    emscripten::constant("CV_64FC2", CV_64FC2) ;
    emscripten::constant("CV_64FC3", CV_64FC3) ;
    emscripten::constant("CV_64FC4", CV_64FC4) ;

    emscripten::constant("CV_8U", CV_8U);
    emscripten::constant("CV_8S", CV_8S);
    emscripten::constant("CV_16U", CV_16U);
    emscripten::constant("CV_16S", CV_16S);
    emscripten::constant("CV_32S",  CV_32S);
    emscripten::constant("CV_32F", CV_32F);
    emscripten::constant("CV_32F", CV_32F);


    emscripten::constant("BORDER_CONSTANT", +cv::BorderTypes::BORDER_CONSTANT);
    emscripten::constant("BORDER_REPLICATE", +cv::BorderTypes::BORDER_REPLICATE);
    emscripten::constant("BORDER_REFLECT", +cv::BorderTypes::BORDER_REFLECT);
    emscripten::constant("BORDER_WRAP", +cv::BorderTypes::BORDER_WRAP);
    emscripten::constant("BORDER_REFLECT_101", +cv::BorderTypes::BORDER_REFLECT_101);
    emscripten::constant("BORDER_TRANSPARENT", +cv::BorderTypes::BORDER_TRANSPARENT);
    emscripten::constant("BORDER_REFLECT101", +cv::BorderTypes::BORDER_REFLECT101);
    emscripten::constant("BORDER_DEFAULT", +cv::BorderTypes::BORDER_DEFAULT);
    emscripten::constant("BORDER_ISOLATED", +cv::BorderTypes::BORDER_ISOLATED);

    emscripten::constant("NORM_INF", +cv::NormTypes::NORM_INF);
    emscripten::constant("NORM_L1", +cv::NormTypes::NORM_L1);
    emscripten::constant("NORM_L2", +cv::NormTypes::NORM_L2);
    emscripten::constant("NORM_L2SQR", +cv::NormTypes::NORM_L2SQR);
    emscripten::constant("NORM_HAMMING", +cv::NormTypes::NORM_HAMMING);
    emscripten::constant("NORM_HAMMING2", +cv::NormTypes::NORM_HAMMING2);
    emscripten::constant("NORM_TYPE_MASK", +cv::NormTypes::NORM_TYPE_MASK);
    emscripten::constant("NORM_RELATIVE", +cv::NormTypes::NORM_RELATIVE);
    emscripten::constant("NORM_MINMAX", +cv::NormTypes::NORM_MINMAX);

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

    void resize_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Size arg3, double arg4, double arg5, int arg6) {
        return cv::resize(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    void sqrt_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::sqrt(arg1, arg2);
    }
    void calcOpticalFlowFarneback_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, double arg4, int arg5, int arg6, int arg7, int arg8, double arg9, int arg10) {
        return cv::calcOpticalFlowFarneback(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }

    std::string Algorithm_getDefaultName_wrapper(cv::Algorithm& arg0 ) {
        return arg0.getDefaultName();
    }

    void Algorithm_save_wrapper(cv::Algorithm& arg0 , const std::string& arg1) {
        return arg0.save(arg1);
    }

    void DenseOpticalFlow_calc_wrapper(cv::DenseOpticalFlow& arg0 , const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return arg0.calc(arg1, arg2, arg3);
    }

    void GaussianBlur_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Size arg3, double arg4, double arg5, int arg6) {
        return cv::GaussianBlur(arg1, arg2, arg3, arg4, arg5, arg6);
    }


}


EMSCRIPTEN_BINDINGS(testBinding) {
    emscripten::function("calcOpticalFlowFarneback", emscripten::select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, double, int, int, int, int, double, int)>(&Wrappers::calcOpticalFlowFarneback_wrapper));

    emscripten::class_<cv::DenseOpticalFlow, emscripten::base<cv::Algorithm>>("DenseOpticalFlow")
        .function("calc", emscripten::select_overload<void(cv::DenseOpticalFlow&,const cv::Mat&,const cv::Mat&,cv::Mat&)>(&Wrappers::DenseOpticalFlow_calc_wrapper), emscripten::pure_virtual())
        .function("collectGarbage", emscripten::select_overload<void()>(&cv::DenseOpticalFlow::collectGarbage), emscripten::pure_virtual());

    emscripten::class_<cv::Algorithm >("Algorithm")
        .function("getDefaultName", emscripten::select_overload<std::string(cv::Algorithm&)>(&Wrappers::Algorithm_getDefaultName_wrapper))
        .function("clear", emscripten::select_overload<void()>(&cv::Algorithm::clear))
        .function("save", emscripten::select_overload<void(cv::Algorithm&,const std::string&)>(&Wrappers::Algorithm_save_wrapper));

    emscripten::function("Canny", emscripten::select_overload<void(const cv::Mat&, cv::Mat&, double, double, int, bool)>(&Wrappers::Canny_wrapper));


    emscripten::function("GaussianBlur", emscripten::select_overload<void(const cv::Mat&, cv::Mat&, cv::Size, double, double, int)>(&Wrappers::GaussianBlur_wrapper));


    emscripten::function("cvtColor", emscripten::select_overload<void(const cv::Mat&, cv::Mat&, int, int)>(&Wrappers::cvtColor_wrapper));

    emscripten::function("integral", emscripten::select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::integral_wrapper));

    emscripten::function("integral2", emscripten::select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, int, int)>(&Wrappers::integral_wrapper));

    emscripten::function("integral3", emscripten::select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&, int, int)>(&Wrappers::integral_wrapper));

    emscripten::function("resize", emscripten::select_overload<void(const cv::Mat&, cv::Mat&, cv::Size, double, double, int)>(&Wrappers::resize_wrapper));

    emscripten::function("sqrt", emscripten::select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::sqrt_wrapper));



    emscripten::enum_<cv::ColorConversionCodes>("ColorConversionCodes")
        .value("COLOR_BGR2BGRA", cv::ColorConversionCodes::COLOR_BGR2BGRA)
        .value("COLOR_RGB2RGBA", cv::ColorConversionCodes::COLOR_RGB2RGBA)
        .value("COLOR_BGRA2BGR", cv::ColorConversionCodes::COLOR_BGRA2BGR)
        .value("COLOR_RGBA2RGB", cv::ColorConversionCodes::COLOR_RGBA2RGB)
        .value("COLOR_BGR2RGBA", cv::ColorConversionCodes::COLOR_BGR2RGBA)
        .value("COLOR_RGB2BGRA", cv::ColorConversionCodes::COLOR_RGB2BGRA)
        .value("COLOR_RGBA2BGR", cv::ColorConversionCodes::COLOR_RGBA2BGR)
        .value("COLOR_BGRA2RGB", cv::ColorConversionCodes::COLOR_BGRA2RGB)
        .value("COLOR_BGR2RGB", cv::ColorConversionCodes::COLOR_BGR2RGB)
        .value("COLOR_RGB2BGR", cv::ColorConversionCodes::COLOR_RGB2BGR)
        .value("COLOR_BGRA2RGBA", cv::ColorConversionCodes::COLOR_BGRA2RGBA)
        .value("COLOR_RGBA2BGRA", cv::ColorConversionCodes::COLOR_RGBA2BGRA)
        .value("COLOR_BGR2GRAY", cv::ColorConversionCodes::COLOR_BGR2GRAY)
        .value("COLOR_RGB2GRAY", cv::ColorConversionCodes::COLOR_RGB2GRAY)
        .value("COLOR_GRAY2BGR", cv::ColorConversionCodes::COLOR_GRAY2BGR)
        .value("COLOR_GRAY2RGB", cv::ColorConversionCodes::COLOR_GRAY2RGB)
        .value("COLOR_GRAY2BGRA", cv::ColorConversionCodes::COLOR_GRAY2BGRA)
        .value("COLOR_GRAY2RGBA", cv::ColorConversionCodes::COLOR_GRAY2RGBA)
        .value("COLOR_BGRA2GRAY", cv::ColorConversionCodes::COLOR_BGRA2GRAY)
        .value("COLOR_RGBA2GRAY", cv::ColorConversionCodes::COLOR_RGBA2GRAY)
        .value("COLOR_BGR2BGR565", cv::ColorConversionCodes::COLOR_BGR2BGR565)
        .value("COLOR_RGB2BGR565", cv::ColorConversionCodes::COLOR_RGB2BGR565)
        .value("COLOR_BGR5652BGR", cv::ColorConversionCodes::COLOR_BGR5652BGR)
        .value("COLOR_BGR5652RGB", cv::ColorConversionCodes::COLOR_BGR5652RGB)
        .value("COLOR_BGRA2BGR565", cv::ColorConversionCodes::COLOR_BGRA2BGR565)
        .value("COLOR_RGBA2BGR565", cv::ColorConversionCodes::COLOR_RGBA2BGR565)
        .value("COLOR_BGR5652BGRA", cv::ColorConversionCodes::COLOR_BGR5652BGRA)
        .value("COLOR_BGR5652RGBA", cv::ColorConversionCodes::COLOR_BGR5652RGBA)
        .value("COLOR_GRAY2BGR565", cv::ColorConversionCodes::COLOR_GRAY2BGR565)
        .value("COLOR_BGR5652GRAY", cv::ColorConversionCodes::COLOR_BGR5652GRAY)
        .value("COLOR_BGR2BGR555", cv::ColorConversionCodes::COLOR_BGR2BGR555)
        .value("COLOR_RGB2BGR555", cv::ColorConversionCodes::COLOR_RGB2BGR555)
        .value("COLOR_BGR5552BGR", cv::ColorConversionCodes::COLOR_BGR5552BGR)
        .value("COLOR_BGR5552RGB", cv::ColorConversionCodes::COLOR_BGR5552RGB)
        .value("COLOR_BGRA2BGR555", cv::ColorConversionCodes::COLOR_BGRA2BGR555)
        .value("COLOR_RGBA2BGR555", cv::ColorConversionCodes::COLOR_RGBA2BGR555)
        .value("COLOR_BGR5552BGRA", cv::ColorConversionCodes::COLOR_BGR5552BGRA)
        .value("COLOR_BGR5552RGBA", cv::ColorConversionCodes::COLOR_BGR5552RGBA)
        .value("COLOR_GRAY2BGR555", cv::ColorConversionCodes::COLOR_GRAY2BGR555)
        .value("COLOR_BGR5552GRAY", cv::ColorConversionCodes::COLOR_BGR5552GRAY)
        .value("COLOR_BGR2XYZ", cv::ColorConversionCodes::COLOR_BGR2XYZ)
        .value("COLOR_RGB2XYZ", cv::ColorConversionCodes::COLOR_RGB2XYZ)
        .value("COLOR_XYZ2BGR", cv::ColorConversionCodes::COLOR_XYZ2BGR)
        .value("COLOR_XYZ2RGB", cv::ColorConversionCodes::COLOR_XYZ2RGB)
        .value("COLOR_BGR2YCrCb", cv::ColorConversionCodes::COLOR_BGR2YCrCb)
        .value("COLOR_RGB2YCrCb", cv::ColorConversionCodes::COLOR_RGB2YCrCb)
        .value("COLOR_YCrCb2BGR", cv::ColorConversionCodes::COLOR_YCrCb2BGR)
        .value("COLOR_YCrCb2RGB", cv::ColorConversionCodes::COLOR_YCrCb2RGB)
        .value("COLOR_BGR2HSV", cv::ColorConversionCodes::COLOR_BGR2HSV)
        .value("COLOR_RGB2HSV", cv::ColorConversionCodes::COLOR_RGB2HSV)
        .value("COLOR_BGR2Lab", cv::ColorConversionCodes::COLOR_BGR2Lab)
        .value("COLOR_RGB2Lab", cv::ColorConversionCodes::COLOR_RGB2Lab)
        .value("COLOR_BGR2Luv", cv::ColorConversionCodes::COLOR_BGR2Luv)
        .value("COLOR_RGB2Luv", cv::ColorConversionCodes::COLOR_RGB2Luv)
        .value("COLOR_BGR2HLS", cv::ColorConversionCodes::COLOR_BGR2HLS)
        .value("COLOR_RGB2HLS", cv::ColorConversionCodes::COLOR_RGB2HLS)
        .value("COLOR_HSV2BGR", cv::ColorConversionCodes::COLOR_HSV2BGR)
        .value("COLOR_HSV2RGB", cv::ColorConversionCodes::COLOR_HSV2RGB)
        .value("COLOR_Lab2BGR", cv::ColorConversionCodes::COLOR_Lab2BGR)
        .value("COLOR_Lab2RGB", cv::ColorConversionCodes::COLOR_Lab2RGB)
        .value("COLOR_Luv2BGR", cv::ColorConversionCodes::COLOR_Luv2BGR)
        .value("COLOR_Luv2RGB", cv::ColorConversionCodes::COLOR_Luv2RGB)
        .value("COLOR_HLS2BGR", cv::ColorConversionCodes::COLOR_HLS2BGR)
        .value("COLOR_HLS2RGB", cv::ColorConversionCodes::COLOR_HLS2RGB)
        .value("COLOR_BGR2HSV_FULL", cv::ColorConversionCodes::COLOR_BGR2HSV_FULL)
        .value("COLOR_RGB2HSV_FULL", cv::ColorConversionCodes::COLOR_RGB2HSV_FULL)
        .value("COLOR_BGR2HLS_FULL", cv::ColorConversionCodes::COLOR_BGR2HLS_FULL)
        .value("COLOR_RGB2HLS_FULL", cv::ColorConversionCodes::COLOR_RGB2HLS_FULL)
        .value("COLOR_HSV2BGR_FULL", cv::ColorConversionCodes::COLOR_HSV2BGR_FULL)
        .value("COLOR_HSV2RGB_FULL", cv::ColorConversionCodes::COLOR_HSV2RGB_FULL)
        .value("COLOR_HLS2BGR_FULL", cv::ColorConversionCodes::COLOR_HLS2BGR_FULL)
        .value("COLOR_HLS2RGB_FULL", cv::ColorConversionCodes::COLOR_HLS2RGB_FULL)
        .value("COLOR_LBGR2Lab", cv::ColorConversionCodes::COLOR_LBGR2Lab)
        .value("COLOR_LRGB2Lab", cv::ColorConversionCodes::COLOR_LRGB2Lab)
        .value("COLOR_LBGR2Luv", cv::ColorConversionCodes::COLOR_LBGR2Luv)
        .value("COLOR_LRGB2Luv", cv::ColorConversionCodes::COLOR_LRGB2Luv)
        .value("COLOR_Lab2LBGR", cv::ColorConversionCodes::COLOR_Lab2LBGR)
        .value("COLOR_Lab2LRGB", cv::ColorConversionCodes::COLOR_Lab2LRGB)
        .value("COLOR_Luv2LBGR", cv::ColorConversionCodes::COLOR_Luv2LBGR)
        .value("COLOR_Luv2LRGB", cv::ColorConversionCodes::COLOR_Luv2LRGB)
        .value("COLOR_BGR2YUV", cv::ColorConversionCodes::COLOR_BGR2YUV)
        .value("COLOR_RGB2YUV", cv::ColorConversionCodes::COLOR_RGB2YUV)
        .value("COLOR_YUV2BGR", cv::ColorConversionCodes::COLOR_YUV2BGR)
        .value("COLOR_YUV2RGB", cv::ColorConversionCodes::COLOR_YUV2RGB)
        .value("COLOR_YUV2RGB_NV12", cv::ColorConversionCodes::COLOR_YUV2RGB_NV12)
        .value("COLOR_YUV2BGR_NV12", cv::ColorConversionCodes::COLOR_YUV2BGR_NV12)
        .value("COLOR_YUV2RGB_NV21", cv::ColorConversionCodes::COLOR_YUV2RGB_NV21)
        .value("COLOR_YUV2BGR_NV21", cv::ColorConversionCodes::COLOR_YUV2BGR_NV21)
        .value("COLOR_YUV420sp2RGB", cv::ColorConversionCodes::COLOR_YUV420sp2RGB)
        .value("COLOR_YUV420sp2BGR", cv::ColorConversionCodes::COLOR_YUV420sp2BGR)
        .value("COLOR_YUV2RGBA_NV12", cv::ColorConversionCodes::COLOR_YUV2RGBA_NV12)
        .value("COLOR_YUV2BGRA_NV12", cv::ColorConversionCodes::COLOR_YUV2BGRA_NV12)
        .value("COLOR_YUV2RGBA_NV21", cv::ColorConversionCodes::COLOR_YUV2RGBA_NV21)
        .value("COLOR_YUV2BGRA_NV21", cv::ColorConversionCodes::COLOR_YUV2BGRA_NV21)
        .value("COLOR_YUV420sp2RGBA", cv::ColorConversionCodes::COLOR_YUV420sp2RGBA)
        .value("COLOR_YUV420sp2BGRA", cv::ColorConversionCodes::COLOR_YUV420sp2BGRA)
        .value("COLOR_YUV2RGB_YV12", cv::ColorConversionCodes::COLOR_YUV2RGB_YV12)
        .value("COLOR_YUV2BGR_YV12", cv::ColorConversionCodes::COLOR_YUV2BGR_YV12)
        .value("COLOR_YUV2RGB_IYUV", cv::ColorConversionCodes::COLOR_YUV2RGB_IYUV)
        .value("COLOR_YUV2BGR_IYUV", cv::ColorConversionCodes::COLOR_YUV2BGR_IYUV)
        .value("COLOR_YUV2RGB_I420", cv::ColorConversionCodes::COLOR_YUV2RGB_I420)
        .value("COLOR_YUV2BGR_I420", cv::ColorConversionCodes::COLOR_YUV2BGR_I420)
        .value("COLOR_YUV420p2RGB", cv::ColorConversionCodes::COLOR_YUV420p2RGB)
        .value("COLOR_YUV420p2BGR", cv::ColorConversionCodes::COLOR_YUV420p2BGR)
        .value("COLOR_YUV2RGBA_YV12", cv::ColorConversionCodes::COLOR_YUV2RGBA_YV12)
        .value("COLOR_YUV2BGRA_YV12", cv::ColorConversionCodes::COLOR_YUV2BGRA_YV12)
        .value("COLOR_YUV2RGBA_IYUV", cv::ColorConversionCodes::COLOR_YUV2RGBA_IYUV)
        .value("COLOR_YUV2BGRA_IYUV", cv::ColorConversionCodes::COLOR_YUV2BGRA_IYUV)
        .value("COLOR_YUV2RGBA_I420", cv::ColorConversionCodes::COLOR_YUV2RGBA_I420)
        .value("COLOR_YUV2BGRA_I420", cv::ColorConversionCodes::COLOR_YUV2BGRA_I420)
        .value("COLOR_YUV420p2RGBA", cv::ColorConversionCodes::COLOR_YUV420p2RGBA)
        .value("COLOR_YUV420p2BGRA", cv::ColorConversionCodes::COLOR_YUV420p2BGRA)
        .value("COLOR_YUV2GRAY_420", cv::ColorConversionCodes::COLOR_YUV2GRAY_420)
        .value("COLOR_YUV2GRAY_NV21", cv::ColorConversionCodes::COLOR_YUV2GRAY_NV21)
        .value("COLOR_YUV2GRAY_NV12", cv::ColorConversionCodes::COLOR_YUV2GRAY_NV12)
        .value("COLOR_YUV2GRAY_YV12", cv::ColorConversionCodes::COLOR_YUV2GRAY_YV12)
        .value("COLOR_YUV2GRAY_IYUV", cv::ColorConversionCodes::COLOR_YUV2GRAY_IYUV)
        .value("COLOR_YUV2GRAY_I420", cv::ColorConversionCodes::COLOR_YUV2GRAY_I420)
        .value("COLOR_YUV420sp2GRAY", cv::ColorConversionCodes::COLOR_YUV420sp2GRAY)
        .value("COLOR_YUV420p2GRAY", cv::ColorConversionCodes::COLOR_YUV420p2GRAY)
        .value("COLOR_YUV2RGB_UYVY", cv::ColorConversionCodes::COLOR_YUV2RGB_UYVY)
        .value("COLOR_YUV2BGR_UYVY", cv::ColorConversionCodes::COLOR_YUV2BGR_UYVY)
        .value("COLOR_YUV2RGB_Y422", cv::ColorConversionCodes::COLOR_YUV2RGB_Y422)
        .value("COLOR_YUV2BGR_Y422", cv::ColorConversionCodes::COLOR_YUV2BGR_Y422)
        .value("COLOR_YUV2RGB_UYNV", cv::ColorConversionCodes::COLOR_YUV2RGB_UYNV)
        .value("COLOR_YUV2BGR_UYNV", cv::ColorConversionCodes::COLOR_YUV2BGR_UYNV)
        .value("COLOR_YUV2RGBA_UYVY", cv::ColorConversionCodes::COLOR_YUV2RGBA_UYVY)
        .value("COLOR_YUV2BGRA_UYVY", cv::ColorConversionCodes::COLOR_YUV2BGRA_UYVY)
        .value("COLOR_YUV2RGBA_Y422", cv::ColorConversionCodes::COLOR_YUV2RGBA_Y422)
        .value("COLOR_YUV2BGRA_Y422", cv::ColorConversionCodes::COLOR_YUV2BGRA_Y422)
        .value("COLOR_YUV2RGBA_UYNV", cv::ColorConversionCodes::COLOR_YUV2RGBA_UYNV)
        .value("COLOR_YUV2BGRA_UYNV", cv::ColorConversionCodes::COLOR_YUV2BGRA_UYNV)
        .value("COLOR_YUV2RGB_YUY2", cv::ColorConversionCodes::COLOR_YUV2RGB_YUY2)
        .value("COLOR_YUV2BGR_YUY2", cv::ColorConversionCodes::COLOR_YUV2BGR_YUY2)
        .value("COLOR_YUV2RGB_YVYU", cv::ColorConversionCodes::COLOR_YUV2RGB_YVYU)
        .value("COLOR_YUV2BGR_YVYU", cv::ColorConversionCodes::COLOR_YUV2BGR_YVYU)
        .value("COLOR_YUV2RGB_YUYV", cv::ColorConversionCodes::COLOR_YUV2RGB_YUYV)
        .value("COLOR_YUV2BGR_YUYV", cv::ColorConversionCodes::COLOR_YUV2BGR_YUYV)
        .value("COLOR_YUV2RGB_YUNV", cv::ColorConversionCodes::COLOR_YUV2RGB_YUNV)
        .value("COLOR_YUV2BGR_YUNV", cv::ColorConversionCodes::COLOR_YUV2BGR_YUNV)
        .value("COLOR_YUV2RGBA_YUY2", cv::ColorConversionCodes::COLOR_YUV2RGBA_YUY2)
        .value("COLOR_YUV2BGRA_YUY2", cv::ColorConversionCodes::COLOR_YUV2BGRA_YUY2)
        .value("COLOR_YUV2RGBA_YVYU", cv::ColorConversionCodes::COLOR_YUV2RGBA_YVYU)
        .value("COLOR_YUV2BGRA_YVYU", cv::ColorConversionCodes::COLOR_YUV2BGRA_YVYU)
        .value("COLOR_YUV2RGBA_YUYV", cv::ColorConversionCodes::COLOR_YUV2RGBA_YUYV)
        .value("COLOR_YUV2BGRA_YUYV", cv::ColorConversionCodes::COLOR_YUV2BGRA_YUYV)
        .value("COLOR_YUV2RGBA_YUNV", cv::ColorConversionCodes::COLOR_YUV2RGBA_YUNV)
        .value("COLOR_YUV2BGRA_YUNV", cv::ColorConversionCodes::COLOR_YUV2BGRA_YUNV)
        .value("COLOR_YUV2GRAY_UYVY", cv::ColorConversionCodes::COLOR_YUV2GRAY_UYVY)
        .value("COLOR_YUV2GRAY_YUY2", cv::ColorConversionCodes::COLOR_YUV2GRAY_YUY2)
        .value("COLOR_YUV2GRAY_Y422", cv::ColorConversionCodes::COLOR_YUV2GRAY_Y422)
        .value("COLOR_YUV2GRAY_UYNV", cv::ColorConversionCodes::COLOR_YUV2GRAY_UYNV)
        .value("COLOR_YUV2GRAY_YVYU", cv::ColorConversionCodes::COLOR_YUV2GRAY_YVYU)
        .value("COLOR_YUV2GRAY_YUYV", cv::ColorConversionCodes::COLOR_YUV2GRAY_YUYV)
        .value("COLOR_YUV2GRAY_YUNV", cv::ColorConversionCodes::COLOR_YUV2GRAY_YUNV)
        .value("COLOR_RGBA2mRGBA", cv::ColorConversionCodes::COLOR_RGBA2mRGBA)
        .value("COLOR_mRGBA2RGBA", cv::ColorConversionCodes::COLOR_mRGBA2RGBA)
        .value("COLOR_RGB2YUV_I420", cv::ColorConversionCodes::COLOR_RGB2YUV_I420)
        .value("COLOR_BGR2YUV_I420", cv::ColorConversionCodes::COLOR_BGR2YUV_I420)
        .value("COLOR_RGB2YUV_IYUV", cv::ColorConversionCodes::COLOR_RGB2YUV_IYUV)
        .value("COLOR_BGR2YUV_IYUV", cv::ColorConversionCodes::COLOR_BGR2YUV_IYUV)
        .value("COLOR_RGBA2YUV_I420", cv::ColorConversionCodes::COLOR_RGBA2YUV_I420)
        .value("COLOR_BGRA2YUV_I420", cv::ColorConversionCodes::COLOR_BGRA2YUV_I420)
        .value("COLOR_RGBA2YUV_IYUV", cv::ColorConversionCodes::COLOR_RGBA2YUV_IYUV)
        .value("COLOR_BGRA2YUV_IYUV", cv::ColorConversionCodes::COLOR_BGRA2YUV_IYUV)
        .value("COLOR_RGB2YUV_YV12", cv::ColorConversionCodes::COLOR_RGB2YUV_YV12)
        .value("COLOR_BGR2YUV_YV12", cv::ColorConversionCodes::COLOR_BGR2YUV_YV12)
        .value("COLOR_RGBA2YUV_YV12", cv::ColorConversionCodes::COLOR_RGBA2YUV_YV12)
        .value("COLOR_BGRA2YUV_YV12", cv::ColorConversionCodes::COLOR_BGRA2YUV_YV12)
        .value("COLOR_BayerBG2BGR", cv::ColorConversionCodes::COLOR_BayerBG2BGR)
        .value("COLOR_BayerGB2BGR", cv::ColorConversionCodes::COLOR_BayerGB2BGR)
        .value("COLOR_BayerRG2BGR", cv::ColorConversionCodes::COLOR_BayerRG2BGR)
        .value("COLOR_BayerGR2BGR", cv::ColorConversionCodes::COLOR_BayerGR2BGR)
        .value("COLOR_BayerBG2RGB", cv::ColorConversionCodes::COLOR_BayerBG2RGB)
        .value("COLOR_BayerGB2RGB", cv::ColorConversionCodes::COLOR_BayerGB2RGB)
        .value("COLOR_BayerRG2RGB", cv::ColorConversionCodes::COLOR_BayerRG2RGB)
        .value("COLOR_BayerGR2RGB", cv::ColorConversionCodes::COLOR_BayerGR2RGB)
        .value("COLOR_BayerBG2GRAY", cv::ColorConversionCodes::COLOR_BayerBG2GRAY)
        .value("COLOR_BayerGB2GRAY", cv::ColorConversionCodes::COLOR_BayerGB2GRAY)
        .value("COLOR_BayerRG2GRAY", cv::ColorConversionCodes::COLOR_BayerRG2GRAY)
        .value("COLOR_BayerGR2GRAY", cv::ColorConversionCodes::COLOR_BayerGR2GRAY)
        .value("COLOR_BayerBG2BGR_VNG", cv::ColorConversionCodes::COLOR_BayerBG2BGR_VNG)
        .value("COLOR_BayerGB2BGR_VNG", cv::ColorConversionCodes::COLOR_BayerGB2BGR_VNG)
        .value("COLOR_BayerRG2BGR_VNG", cv::ColorConversionCodes::COLOR_BayerRG2BGR_VNG)
        .value("COLOR_BayerGR2BGR_VNG", cv::ColorConversionCodes::COLOR_BayerGR2BGR_VNG)
        .value("COLOR_BayerBG2RGB_VNG", cv::ColorConversionCodes::COLOR_BayerBG2RGB_VNG)
        .value("COLOR_BayerGB2RGB_VNG", cv::ColorConversionCodes::COLOR_BayerGB2RGB_VNG)
        .value("COLOR_BayerRG2RGB_VNG", cv::ColorConversionCodes::COLOR_BayerRG2RGB_VNG)
        .value("COLOR_BayerGR2RGB_VNG", cv::ColorConversionCodes::COLOR_BayerGR2RGB_VNG)
        .value("COLOR_BayerBG2BGR_EA", cv::ColorConversionCodes::COLOR_BayerBG2BGR_EA)
        .value("COLOR_BayerGB2BGR_EA", cv::ColorConversionCodes::COLOR_BayerGB2BGR_EA)
        .value("COLOR_BayerRG2BGR_EA", cv::ColorConversionCodes::COLOR_BayerRG2BGR_EA)
        .value("COLOR_BayerGR2BGR_EA", cv::ColorConversionCodes::COLOR_BayerGR2BGR_EA)
        .value("COLOR_BayerBG2RGB_EA", cv::ColorConversionCodes::COLOR_BayerBG2RGB_EA)
        .value("COLOR_BayerGB2RGB_EA", cv::ColorConversionCodes::COLOR_BayerGB2RGB_EA)
        .value("COLOR_BayerRG2RGB_EA", cv::ColorConversionCodes::COLOR_BayerRG2RGB_EA)
        .value("COLOR_BayerGR2RGB_EA", cv::ColorConversionCodes::COLOR_BayerGR2RGB_EA)
        .value("COLOR_COLORCVT_MAX", cv::ColorConversionCodes::COLOR_COLORCVT_MAX);

    emscripten::enum_<cv::ColormapTypes>("ColormapTypes")
        .value("COLORMAP_AUTUMN", cv::ColormapTypes::COLORMAP_AUTUMN)
        .value("COLORMAP_BONE", cv::ColormapTypes::COLORMAP_BONE)
        .value("COLORMAP_JET", cv::ColormapTypes::COLORMAP_JET)
        .value("COLORMAP_WINTER", cv::ColormapTypes::COLORMAP_WINTER)
        .value("COLORMAP_RAINBOW", cv::ColormapTypes::COLORMAP_RAINBOW)
        .value("COLORMAP_OCEAN", cv::ColormapTypes::COLORMAP_OCEAN)
        .value("COLORMAP_SUMMER", cv::ColormapTypes::COLORMAP_SUMMER)
        .value("COLORMAP_SPRING", cv::ColormapTypes::COLORMAP_SPRING)
        .value("COLORMAP_COOL", cv::ColormapTypes::COLORMAP_COOL)
        .value("COLORMAP_HSV", cv::ColormapTypes::COLORMAP_HSV)
        .value("COLORMAP_PINK", cv::ColormapTypes::COLORMAP_PINK)
        .value("COLORMAP_HOT", cv::ColormapTypes::COLORMAP_HOT)
        .value("COLORMAP_PARULA", cv::ColormapTypes::COLORMAP_PARULA);

    emscripten::enum_<cv::ConnectedComponentsTypes>("ConnectedComponentsTypes")
        .value("CC_STAT_LEFT", cv::ConnectedComponentsTypes::CC_STAT_LEFT)
        .value("CC_STAT_TOP", cv::ConnectedComponentsTypes::CC_STAT_TOP)
        .value("CC_STAT_WIDTH", cv::ConnectedComponentsTypes::CC_STAT_WIDTH)
        .value("CC_STAT_HEIGHT", cv::ConnectedComponentsTypes::CC_STAT_HEIGHT)
        .value("CC_STAT_AREA", cv::ConnectedComponentsTypes::CC_STAT_AREA)
        .value("CC_STAT_MAX", cv::ConnectedComponentsTypes::CC_STAT_MAX);

    emscripten::enum_<cv::ContourApproximationModes>("ContourApproximationModes")
        .value("CHAIN_APPROX_NONE", cv::ContourApproximationModes::CHAIN_APPROX_NONE)
        .value("CHAIN_APPROX_SIMPLE", cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE)
        .value("CHAIN_APPROX_TC89_L1", cv::ContourApproximationModes::CHAIN_APPROX_TC89_L1)
        .value("CHAIN_APPROX_TC89_KCOS", cv::ContourApproximationModes::CHAIN_APPROX_TC89_KCOS);

    emscripten::enum_<cv::CovarFlags>("CovarFlags")
        .value("COVAR_SCRAMBLED", cv::CovarFlags::COVAR_SCRAMBLED)
        .value("COVAR_NORMAL", cv::CovarFlags::COVAR_NORMAL)
        .value("COVAR_USE_AVG", cv::CovarFlags::COVAR_USE_AVG)
        .value("COVAR_SCALE", cv::CovarFlags::COVAR_SCALE)
        .value("COVAR_ROWS", cv::CovarFlags::COVAR_ROWS)
        .value("COVAR_COLS", cv::CovarFlags::COVAR_COLS);

    emscripten::enum_<cv::DistanceTransformLabelTypes>("DistanceTransformLabelTypes")
        .value("DIST_LABEL_CCOMP", cv::DistanceTransformLabelTypes::DIST_LABEL_CCOMP)
        .value("DIST_LABEL_PIXEL", cv::DistanceTransformLabelTypes::DIST_LABEL_PIXEL);

    emscripten::enum_<cv::DistanceTransformMasks>("DistanceTransformMasks")
        .value("DIST_MASK_3", cv::DistanceTransformMasks::DIST_MASK_3)
        .value("DIST_MASK_5", cv::DistanceTransformMasks::DIST_MASK_5)
        .value("DIST_MASK_PRECISE", cv::DistanceTransformMasks::DIST_MASK_PRECISE);

    emscripten::enum_<cv::DistanceTypes>("DistanceTypes")
        .value("DIST_USER", cv::DistanceTypes::DIST_USER)
        .value("DIST_L1", cv::DistanceTypes::DIST_L1)
        .value("DIST_L2", cv::DistanceTypes::DIST_L2)
        .value("DIST_C", cv::DistanceTypes::DIST_C)
        .value("DIST_L12", cv::DistanceTypes::DIST_L12)
        .value("DIST_FAIR", cv::DistanceTypes::DIST_FAIR)
        .value("DIST_WELSCH", cv::DistanceTypes::DIST_WELSCH)
        .value("DIST_HUBER", cv::DistanceTypes::DIST_HUBER);

    emscripten::enum_<cv::FloodFillFlags>("FloodFillFlags")
        .value("FLOODFILL_FIXED_RANGE", cv::FloodFillFlags::FLOODFILL_FIXED_RANGE)
        .value("FLOODFILL_MASK_ONLY", cv::FloodFillFlags::FLOODFILL_MASK_ONLY);

    emscripten::enum_<cv::GrabCutClasses>("GrabCutClasses")
        .value("GC_BGD", cv::GrabCutClasses::GC_BGD)
        .value("GC_FGD", cv::GrabCutClasses::GC_FGD)
        .value("GC_PR_BGD", cv::GrabCutClasses::GC_PR_BGD)
        .value("GC_PR_FGD", cv::GrabCutClasses::GC_PR_FGD);

    emscripten::enum_<cv::GrabCutModes>("GrabCutModes")
        .value("GC_INIT_WITH_RECT", cv::GrabCutModes::GC_INIT_WITH_RECT)
        .value("GC_INIT_WITH_MASK", cv::GrabCutModes::GC_INIT_WITH_MASK)
        .value("GC_EVAL", cv::GrabCutModes::GC_EVAL);

    emscripten::enum_<cv::HersheyFonts>("HersheyFonts")
        .value("FONT_HERSHEY_SIMPLEX", cv::HersheyFonts::FONT_HERSHEY_SIMPLEX)
        .value("FONT_HERSHEY_PLAIN", cv::HersheyFonts::FONT_HERSHEY_PLAIN)
        .value("FONT_HERSHEY_DUPLEX", cv::HersheyFonts::FONT_HERSHEY_DUPLEX)
        .value("FONT_HERSHEY_COMPLEX", cv::HersheyFonts::FONT_HERSHEY_COMPLEX)
        .value("FONT_HERSHEY_TRIPLEX", cv::HersheyFonts::FONT_HERSHEY_TRIPLEX)
        .value("FONT_HERSHEY_COMPLEX_SMALL", cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL)
        .value("FONT_HERSHEY_SCRIPT_SIMPLEX", cv::HersheyFonts::FONT_HERSHEY_SCRIPT_SIMPLEX)
        .value("FONT_HERSHEY_SCRIPT_COMPLEX", cv::HersheyFonts::FONT_HERSHEY_SCRIPT_COMPLEX)
        .value("FONT_ITALIC", cv::HersheyFonts::FONT_ITALIC);

    emscripten::enum_<cv::HistCompMethods>("HistCompMethods")
        .value("HISTCMP_CORREL", cv::HistCompMethods::HISTCMP_CORREL)
        .value("HISTCMP_CHISQR", cv::HistCompMethods::HISTCMP_CHISQR)
        .value("HISTCMP_INTERSECT", cv::HistCompMethods::HISTCMP_INTERSECT)
        .value("HISTCMP_BHATTACHARYYA", cv::HistCompMethods::HISTCMP_BHATTACHARYYA)
        .value("HISTCMP_HELLINGER", cv::HistCompMethods::HISTCMP_HELLINGER)
        .value("HISTCMP_CHISQR_ALT", cv::HistCompMethods::HISTCMP_CHISQR_ALT)
        .value("HISTCMP_KL_DIV", cv::HistCompMethods::HISTCMP_KL_DIV);

    emscripten::enum_<cv::HoughModes>("HoughModes")
        .value("HOUGH_STANDARD", cv::HoughModes::HOUGH_STANDARD)
        .value("HOUGH_PROBABILISTIC", cv::HoughModes::HOUGH_PROBABILISTIC)
        .value("HOUGH_MULTI_SCALE", cv::HoughModes::HOUGH_MULTI_SCALE)
        .value("HOUGH_GRADIENT", cv::HoughModes::HOUGH_GRADIENT);

    emscripten::enum_<cv::InterpolationFlags>("InterpolationFlags")
        .value("INTER_NEAREST", cv::InterpolationFlags::INTER_NEAREST)
        .value("INTER_LINEAR", cv::InterpolationFlags::INTER_LINEAR)
        .value("INTER_CUBIC", cv::InterpolationFlags::INTER_CUBIC)
        .value("INTER_AREA", cv::InterpolationFlags::INTER_AREA)
        .value("INTER_LANCZOS4", cv::InterpolationFlags::INTER_LANCZOS4)
        .value("INTER_MAX", cv::InterpolationFlags::INTER_MAX)
        .value("WARP_FILL_OUTLIERS", cv::InterpolationFlags::WARP_FILL_OUTLIERS)
        .value("WARP_INVERSE_MAP", cv::InterpolationFlags::WARP_INVERSE_MAP);

    emscripten::enum_<cv::InterpolationMasks>("InterpolationMasks")
        .value("INTER_BITS", cv::InterpolationMasks::INTER_BITS)
        .value("INTER_BITS2", cv::InterpolationMasks::INTER_BITS2)
        .value("INTER_TAB_SIZE", cv::InterpolationMasks::INTER_TAB_SIZE)
        .value("INTER_TAB_SIZE2", cv::InterpolationMasks::INTER_TAB_SIZE2);

    emscripten::enum_<cv::KmeansFlags>("KmeansFlags")
        .value("KMEANS_RANDOM_CENTERS", cv::KmeansFlags::KMEANS_RANDOM_CENTERS)
        .value("KMEANS_PP_CENTERS", cv::KmeansFlags::KMEANS_PP_CENTERS)
        .value("KMEANS_USE_INITIAL_LABELS", cv::KmeansFlags::KMEANS_USE_INITIAL_LABELS);

    emscripten::enum_<cv::LineSegmentDetectorModes>("LineSegmentDetectorModes")
        .value("LSD_REFINE_NONE", cv::LineSegmentDetectorModes::LSD_REFINE_NONE)
        .value("LSD_REFINE_STD", cv::LineSegmentDetectorModes::LSD_REFINE_STD)
        .value("LSD_REFINE_ADV", cv::LineSegmentDetectorModes::LSD_REFINE_ADV);

    emscripten::enum_<cv::LineTypes>("LineTypes")
        .value("FILLED", cv::LineTypes::FILLED)
        .value("LINE_4", cv::LineTypes::LINE_4)
        .value("LINE_8", cv::LineTypes::LINE_8)
        .value("LINE_AA", cv::LineTypes::LINE_AA);

    emscripten::enum_<cv::MarkerTypes>("MarkerTypes")
        .value("MARKER_CROSS", cv::MarkerTypes::MARKER_CROSS)
        .value("MARKER_TILTED_CROSS", cv::MarkerTypes::MARKER_TILTED_CROSS)
        .value("MARKER_STAR", cv::MarkerTypes::MARKER_STAR)
        .value("MARKER_DIAMOND", cv::MarkerTypes::MARKER_DIAMOND)
        .value("MARKER_SQUARE", cv::MarkerTypes::MARKER_SQUARE)
        .value("MARKER_TRIANGLE_UP", cv::MarkerTypes::MARKER_TRIANGLE_UP)
        .value("MARKER_TRIANGLE_DOWN", cv::MarkerTypes::MARKER_TRIANGLE_DOWN);

    emscripten::enum_<cv::MorphShapes>("MorphShapes")
        .value("MORPH_RECT", cv::MorphShapes::MORPH_RECT)
        .value("MORPH_CROSS", cv::MorphShapes::MORPH_CROSS)
        .value("MORPH_ELLIPSE", cv::MorphShapes::MORPH_ELLIPSE);

    emscripten::enum_<cv::MorphTypes>("MorphTypes")
        .value("MORPH_ERODE", cv::MorphTypes::MORPH_ERODE)
        .value("MORPH_DILATE", cv::MorphTypes::MORPH_DILATE)
        .value("MORPH_OPEN", cv::MorphTypes::MORPH_OPEN)
        .value("MORPH_CLOSE", cv::MorphTypes::MORPH_CLOSE)
        .value("MORPH_GRADIENT", cv::MorphTypes::MORPH_GRADIENT)
        .value("MORPH_TOPHAT", cv::MorphTypes::MORPH_TOPHAT)
        .value("MORPH_BLACKHAT", cv::MorphTypes::MORPH_BLACKHAT)
        .value("MORPH_HITMISS", cv::MorphTypes::MORPH_HITMISS);

    emscripten::enum_<cv::PCA::Flags>("PCA_Flags")
        .value("DATA_AS_ROW", cv::PCA::Flags::DATA_AS_ROW)
        .value("DATA_AS_COL", cv::PCA::Flags::DATA_AS_COL)
        .value("USE_AVG", cv::PCA::Flags::USE_AVG);

    emscripten::enum_<cv::RectanglesIntersectTypes>("RectanglesIntersectTypes")
        .value("INTERSECT_NONE", cv::RectanglesIntersectTypes::INTERSECT_NONE)
        .value("INTERSECT_PARTIAL", cv::RectanglesIntersectTypes::INTERSECT_PARTIAL)
        .value("INTERSECT_FULL", cv::RectanglesIntersectTypes::INTERSECT_FULL);

    emscripten::enum_<cv::ReduceTypes>("ReduceTypes")
        .value("REDUCE_SUM", cv::ReduceTypes::REDUCE_SUM)
        .value("REDUCE_AVG", cv::ReduceTypes::REDUCE_AVG)
        .value("REDUCE_MAX", cv::ReduceTypes::REDUCE_MAX)
        .value("REDUCE_MIN", cv::ReduceTypes::REDUCE_MIN);

    emscripten::enum_<cv::RetrievalModes>("RetrievalModes")
        .value("RETR_EXTERNAL", cv::RetrievalModes::RETR_EXTERNAL)
        .value("RETR_LIST", cv::RetrievalModes::RETR_LIST)
        .value("RETR_CCOMP", cv::RetrievalModes::RETR_CCOMP)
        .value("RETR_TREE", cv::RetrievalModes::RETR_TREE)
        .value("RETR_FLOODFILL", cv::RetrievalModes::RETR_FLOODFILL);

    emscripten::enum_<cv::SVD::Flags>("SVD_Flags")
        .value("MODIFY_A", cv::SVD::Flags::MODIFY_A)
        .value("NO_UV", cv::SVD::Flags::NO_UV)
        .value("FULL_UV", cv::SVD::Flags::FULL_UV);

    emscripten::enum_<cv::SortFlags>("SortFlags")
        .value("SORT_EVERY_ROW", cv::SortFlags::SORT_EVERY_ROW)
        .value("SORT_EVERY_COLUMN", cv::SortFlags::SORT_EVERY_COLUMN)
        .value("SORT_ASCENDING", cv::SortFlags::SORT_ASCENDING)
        .value("SORT_DESCENDING", cv::SortFlags::SORT_DESCENDING);

    emscripten::enum_<cv::TemplateMatchModes>("TemplateMatchModes")
        .value("TM_SQDIFF", cv::TemplateMatchModes::TM_SQDIFF)
        .value("TM_SQDIFF_NORMED", cv::TemplateMatchModes::TM_SQDIFF_NORMED)
        .value("TM_CCORR", cv::TemplateMatchModes::TM_CCORR)
        .value("TM_CCORR_NORMED", cv::TemplateMatchModes::TM_CCORR_NORMED)
        .value("TM_CCOEFF", cv::TemplateMatchModes::TM_CCOEFF)
        .value("TM_CCOEFF_NORMED", cv::TemplateMatchModes::TM_CCOEFF_NORMED);

    emscripten::enum_<cv::TermCriteria::Type>("TermCriteria_Type")
        .value("COUNT", cv::TermCriteria::Type::COUNT)
        .value("MAX_ITER", cv::TermCriteria::Type::MAX_ITER)
        .value("EPS", cv::TermCriteria::Type::EPS);

    emscripten::enum_<cv::ThresholdTypes>("ThresholdTypes")
        .value("THRESH_BINARY", cv::ThresholdTypes::THRESH_BINARY)
        .value("THRESH_BINARY_INV", cv::ThresholdTypes::THRESH_BINARY_INV)
        .value("THRESH_TRUNC", cv::ThresholdTypes::THRESH_TRUNC)
        .value("THRESH_TOZERO", cv::ThresholdTypes::THRESH_TOZERO)
        .value("THRESH_TOZERO_INV", cv::ThresholdTypes::THRESH_TOZERO_INV)
        .value("THRESH_MASK", cv::ThresholdTypes::THRESH_MASK)
        .value("THRESH_OTSU", cv::ThresholdTypes::THRESH_OTSU)
        .value("THRESH_TRIANGLE", cv::ThresholdTypes::THRESH_TRIANGLE);

    emscripten::enum_<cv::UMatUsageFlags>("UMatUsageFlags")
        .value("USAGE_DEFAULT", cv::UMatUsageFlags::USAGE_DEFAULT)
        .value("USAGE_ALLOCATE_HOST_MEMORY", cv::UMatUsageFlags::USAGE_ALLOCATE_HOST_MEMORY)
        .value("USAGE_ALLOCATE_DEVICE_MEMORY", cv::UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY)
        .value("USAGE_ALLOCATE_SHARED_MEMORY", cv::UMatUsageFlags::USAGE_ALLOCATE_SHARED_MEMORY)
        .value("__UMAT_USAGE_FLAGS_32BIT", cv::UMatUsageFlags::__UMAT_USAGE_FLAGS_32BIT);

    emscripten::enum_<cv::UndistortTypes>("UndistortTypes")
        .value("PROJ_SPHERICAL_ORTHO", cv::UndistortTypes::PROJ_SPHERICAL_ORTHO)
        .value("PROJ_SPHERICAL_EQRECT", cv::UndistortTypes::PROJ_SPHERICAL_EQRECT);

}