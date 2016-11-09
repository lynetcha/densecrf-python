
#include "densecrf.h"
#include "optimization.h"
#include <cstdio>
#include <cmath>
#include <iostream>
#include <cstdlib>


// The energy object implements an energy function that is minimized using LBFGS
class CRFEnergy: public EnergyFunction {
protected:
	VectorXf initial_u_param_, initial_lbl_param_, initial_knl_param_;
	DenseCRF & crf_;
	const ObjectiveFunction & objective_;
	int NIT_;
	bool unary_, pairwise_, kernel_;
	float l2_norm_;
public:
	CRFEnergy( DenseCRF & crf, const ObjectiveFunction & objective, int NIT, bool unary=1, bool pairwise=1, bool kernel=1 ):crf_(crf),objective_(objective),NIT_(NIT),unary_(unary),pairwise_(pairwise),kernel_(kernel),l2_norm_(0.f){
		initial_u_param_ = crf_.unaryParameters();
		initial_lbl_param_ = crf_.labelCompatibilityParameters();
		initial_knl_param_ = crf_.kernelParameters();
	}
	void setL2Norm( float norm ) {
		l2_norm_ = norm;
	}
	virtual VectorXf initialValue() {
		VectorXf p( unary_*initial_u_param_.rows() + pairwise_*initial_lbl_param_.rows() + kernel_*initial_knl_param_.rows() );
		p << (unary_?initial_u_param_:VectorXf()), (pairwise_?initial_lbl_param_:VectorXf()), (kernel_?initial_knl_param_:VectorXf());
		return p;
	}
	virtual double gradient( const VectorXf & x, VectorXf & dx ) {
		int p = 0;
		if (unary_) {
			crf_.setUnaryParameters( x.segment( p, initial_u_param_.rows() ) );
			p += initial_u_param_.rows();
		}
		if (pairwise_) {
			crf_.setLabelCompatibilityParameters( x.segment( p, initial_lbl_param_.rows() ) );
			p += initial_lbl_param_.rows();
		}
		if (kernel_)
			crf_.setKernelParameters( x.segment( p, initial_knl_param_.rows() ) );
		
		VectorXf du = 0*initial_u_param_, dl = 0*initial_u_param_, dk = 0*initial_knl_param_;
		double r = crf_.gradient( NIT_, objective_, unary_?&du:NULL, pairwise_?&dl:NULL, kernel_?&dk:NULL );
		dx.resize( unary_*du.rows() + pairwise_*dl.rows() + kernel_*dk.rows() );
		dx << -(unary_?du:VectorXf()), -(pairwise_?dl:VectorXf()), -(kernel_?dk:VectorXf());
		r = -r;
		if( l2_norm_ > 0 ) {
			dx += l2_norm_ * x;
			r += 0.5*l2_norm_ * (x.dot(x));
		}
		
		return r;
	}
};

