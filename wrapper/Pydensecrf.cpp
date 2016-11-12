#include <iostream>
#include <boost/python.hpp>
#include <boost/cstdint.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <boost/python/extract.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/numpy.hpp>
#include <Eigen/Eigen>
#include "eigen_numpy.h"
#include "densecrf.h"
#include "objective.h"
#include "optimization.h"
#include "labelcompatibility.h"
#include "crfenergy.h"
#include "pairwise.h"
#include "common.h"

namespace py=boost::python;
namespace np=boost::numpy;

void (DenseCRF::*addPairwiseEnergy)(const Eigen::MatrixXf &,PottsCompatibility *,KernelType,NormalizationType) =  &DenseCRF::addPairwiseEnergy;
void (DenseCRF2D::*addPairwiseEnergy2D)(const Eigen::MatrixXf &,PottsCompatibility *,KernelType,NormalizationType) =  &DenseCRF2D::addPairwiseEnergy;

void (DenseCRF2D::*addPairwiseGaussianPotts)( float , float , PottsCompatibility * , KernelType , NormalizationType ) = &DenseCRF2D::addPairwiseGaussian;
void (DenseCRF2D::*addPairwiseGaussianMatrix)( float , float , MatrixCompatibility * , KernelType , NormalizationType ) = &DenseCRF2D::addPairwiseGaussian;

void (DenseCRF2D::*addPairwiseBilateralPotts)( float , float ,  float , float, float, const unsigned char *,PottsCompatibility * , KernelType , NormalizationType ) = &DenseCRF2D::addPairwiseBilateral;
void (DenseCRF2D::*addPairwiseBilateralMatrix)( float , float ,  float , float,float, const unsigned char*, MatrixCompatibility * , KernelType , NormalizationType ) = &DenseCRF2D::addPairwiseBilateral;

void (DenseCRF::*setUnaryEnergy)(const Eigen::MatrixXf &) = &DenseCRF::setUnaryEnergy;
void (DenseCRF::*setUnaryEnergyLogistic)(const Eigen::MatrixXf &,const Eigen::MatrixXf &) = &DenseCRF::setUnaryEnergy;

void (DenseCRF2D::*setUnaryEnergy2D)(const Eigen::MatrixXf &) = &DenseCRF2D::setUnaryEnergy;
void (DenseCRF2D::*setUnaryEnergyLogistic2D)(const Eigen::MatrixXf &,const Eigen::MatrixXf &) = &DenseCRF2D::setUnaryEnergy;

static const int X = Eigen::Dynamic;


template <typename CType>
py::object addPairwiseBilateralWrapper(py::tuple args, py::dict kwargs)
{
    DenseCRF2D& dcrf = py::extract<DenseCRF2D&>(args[0]);
    np::ndarray img = py::extract<np::ndarray>(args[6]);
    uint8_t * img_data = reinterpret_cast<uint8_t*>(img.get_data());

    dcrf.addPairwiseBilateral(py::extract<float>(args[1]),
                              py::extract<float>(args[2]),
                              py::extract<float>(args[3]),
                              py::extract<float>(args[4]),
                              py::extract<float>(args[5]),
                              img_data,
                              py::extract<CType*>(args[7]),
                              py::extract<KernelType>(args[8]),
                              py::extract<NormalizationType>(args[9]));

    return py::object();
}

NormalizationType identityNorm_(NormalizationType norm) { return norm;}

KernelType identityKernel_(KernelType k) { return k;}

BOOST_PYTHON_MODULE(Pydensecrf)
{
    boost::numpy::initialize();
    SetupEigenConverters();

    py::class_<DenseCRF>("DenseCRF", py::init<int, int>())
        .def("addPairwiseEnergy", addPairwiseEnergy)
        .def("setUnaryEnergy", setUnaryEnergy)
        .def("setUnaryEnergy", setUnaryEnergyLogistic)
        .def("inference", &DenseCRF::inference)
        .def("startInference", &DenseCRF::startInference)
        .def("stepInference", &DenseCRF::stepInference)
        .def("gradient", &DenseCRF::gradient)
        .def("unaryEnergy", &DenseCRF::unaryEnergy)
        .def("pairwiseEnergy", &DenseCRF::pairwiseEnergy)
        .def("klDivergence", &DenseCRF::klDivergence)
        .def("unaryParameters",&DenseCRF::unaryParameters)
        .def("setUnaryParameters",&DenseCRF::setUnaryParameters)
        .def("labelCompatibilityParameters", &DenseCRF::labelCompatibilityParameters)
        .def("setLabelCompatibilityParameters", &DenseCRF::setLabelCompatibilityParameters)
        .def("kernelParameters",&DenseCRF::kernelParameters)
        .def("setKernelParameters",&DenseCRF::setKernelParameters)
        ;

    py::class_<DenseCRF2D,py::bases<DenseCRF> >("DenseCRF2D", py::init<int, int,int>())
        .def("addPairwiseGaussian", addPairwiseGaussianPotts)
        .def("addPairwiseGaussian", addPairwiseGaussianMatrix)
        .def("addPairwiseBilateral",py::raw_function(&addPairwiseBilateralWrapper<PottsCompatibility>, 10))
        .def("addPairwiseBilateral", py::raw_function(&addPairwiseBilateralWrapper<MatrixCompatibility>, 10))
        .def("setUnaryEnergy", setUnaryEnergy2D)
        .def("setUnaryEnergy", setUnaryEnergyLogistic2D)
        .def("addPairwiseEnergy", addPairwiseEnergy2D)
        ;

    py::class_<EnergyFunction,boost::noncopyable>("EnergyFunction",py::no_init);

    py::class_<CRFEnergy,py::bases<EnergyFunction> >(
            "CRFEnergy",py::init<DenseCRF &,const ObjectiveFunction &, int, bool,bool,bool>())
        .def("setL2Norm",&CRFEnergy::setL2Norm)
        .def("gradient",&CRFEnergy::gradient);

    py::def("minimizeLBFGS",&minimizeLBFGS);
    py::def("numericGradient",&numericGradient);
    py::def("gradient",&gradient);
    py::def("gradCheck",&gradCheck);
    py::def("computeFunction",&computeFunction);

    py::enum_<NormalizationType>("NormalizationType")
        .value("NO_NORMALIZATION", NO_NORMALIZATION)
        .value("NORMALIZE_BEFORE",NORMALIZE_BEFORE)
        .value("NORMALIZE_AFTER",NORMALIZE_AFTER)
        .value("NORMALIZE_SYMMETRIC",NORMALIZE_SYMMETRIC);
    py::def("identityNorm",identityNorm_);

    py::enum_<KernelType>("KernelType")
        .value("CONST_KERNEL",CONST_KERNEL)
        .value("DIAG_KERNEL",DIAG_KERNEL)
        .value("FULL_KERNEL",FULL_KERNEL);
    py::def("identityKernel",identityKernel_);

    py::class_<LabelCompatibility,boost::noncopyable>("LabelCompatibility",py::no_init);

    py::class_<PottsCompatibility, py::bases<LabelCompatibility> >("PottsCompatibility",py::init<int>())
        .def("parameters",&PottsCompatibility::parameters);

    py::class_<MatrixCompatibility, py::bases<LabelCompatibility> >("MatrixCompatibility",py::init<const MatrixXf &>())
        .def("parameters",&MatrixCompatibility::parameters);

    py::class_<ObjectiveFunction,boost::noncopyable>("ObjectiveFunction",py::no_init);

    py::class_<IntersectionOverUnion, py::bases<ObjectiveFunction> >("IntersectionOverUnion",py::init<const VectorXs &>())
        .def("evaluate", &IntersectionOverUnion::evaluate);
    py::class_<Hamming, py::bases<ObjectiveFunction> >("Hamming",py::init<const VectorXs &,const VectorXf &>())
        .def("evaluate", &Hamming::evaluate);

    import_array();
}
