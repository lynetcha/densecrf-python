#include <boost/python.hpp>
#include <boost/python/init.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/raw_function.hpp>
#include "densecrf.h"
#include "objective.h"
#include "optimization.h"
#include "crfenergy.h"

namespace py=boost::python;

void (DenseCRF::*addPairwiseEnergy)(const MatrixXf &,LabelCompatibility *,KernelType,NormalizationType) =  &DenseCRF::addPairwiseEnergy;

void (DenseCRF::*setUnaryEnergy)(const MatrixXf &, const MatrixXf &) = &DenseCRF::setUnaryEnergy;


BOOST_PYTHON_MODULE(Pydensecrf)
{
    py::class_<DenseCRF>("DenseCRF", py::init<int, int>())
        .def("addPairwiseEnergy", addPairwiseEnergy)
        .def("setUnaryEnergy", setUnaryEnergy)
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

    //py::class_<CRFEnergy>("CRFEnergy")
    //    .def("setL2Norm",&CRFEnergy::setL2Norm)
    //    .def("gradient",&CRFEnergy::gradient);

    //py::def("minimizeLBFGS",&minimizeLBFGS);
    //py::def("numericGradient",&numericGradient);
    //py::def("gradient",&gradient);
    //py::def("gradCheck",&gradCheck);
    //py::def("computeFunction",&computeFunction);
}
