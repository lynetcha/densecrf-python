#include <iostream>
#include <boost/python.hpp>
#include <boost/python/init.hpp>
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
namespace py=boost::python;

void (DenseCRF::*addPairwiseEnergy)(const Eigen::MatrixXf &,PottsCompatibility *,KernelType,NormalizationType) =  &DenseCRF::addPairwiseEnergy;

void (DenseCRF::*setUnaryEnergy)(const Eigen::MatrixXf &) = &DenseCRF::setUnaryEnergy;

static const int X = Eigen::Dynamic;

void PrintMat(const DenseCRF & self, const Matrix3f& m) {
    std::cout << "Matrix : " << m << std::endl;
}

NormalizationType identityNorm_(NormalizationType norm) { return norm;}

KernelType identityKernel_(KernelType k) { return k;}
// py::object unary_wrapper(py::tuple args, py::dict kwargs)
// {
//     DenseCRF& dcrf = py::extract<DenseCRF&>(args[0]);
//     const Eigen::Matrix<double,X,X> & m = args[1];
//     // dcrf.setUnaryEnergy(m);
// 
//     return py::object();
// }

BOOST_PYTHON_MODULE(Pydensecrf)
{
    boost::numpy::initialize();
    SetupEigenConverters();

    py::class_<DenseCRF>("DenseCRF", py::init<int, int>())
        .def("addPairwiseEnergy", addPairwiseEnergy)
        .def("printMat", PrintMat)
        .def("setUnaryEnergy", setUnaryEnergy)
        // .def("setUnaryEnergy", py::raw_function(&unary_wrapper, 2))
        .def("inference", &DenseCRF::inference)
        .def("startInference", &DenseCRF::startInference)
        .def("stepInference", &DenseCRF::stepInference)
        .def("gradient", &DenseCRF::gradient)
        .def("unaryEnergy", &DenseCRF::unaryEnergy)
        .def("pairwiseEnergy", &DenseCRF::pairwiseEnergy)
        .def("klDivergence", &DenseCRF::klDivergence);


    //py::class_<EnergyFunction>("EnergyFunction")
    //   .def("gradient",&EnergyFunction::gradient);


    //py::class_<EnergyFunction>("ObjectiveFunction")
    //    .def("evaluate",&ObjectiveFunction::evaluate)
    //    .def("gradient",&ObjectiveFunction::gradient);


    //py::class_<EnergyFunction>("LogLikelihood")
    //    .def("evaluate",&LogLikelihood::evaluate)
    //    .def("gradient",&LogLikelihood::gradient);

    py::class_<CRFEnergy>("CRFEnergy",py::init<DenseCRF &,const ObjectiveFunction &, int, bool,bool,bool>())
        .def("setL2Norm",&CRFEnergy::setL2Norm)
        .def("gradient",&CRFEnergy::gradient);

    //py::def("minimizeLBFGS",&minimizeLBFGS);
    //py::def("numericGradient",&numericGradient);
    //py::def("gradient",&gradient);
    //py::def("gradCheck",&gradCheck);
    //py::def("computeFunction",&computeFunction);

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


}
