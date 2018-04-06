#ifndef _ELASTICNETREG_H_
#define _ELASTICNETREG_H_

#include "ltt/vector.hpp"
#include "AFL/PAL/Utilities/Matrix/Matrix.hpp"
#include "AFL/PAL/Utilities/Matrix/DenseSparseMatrix.hpp"

namespace AFL_PAL
{
namespace LOGR
{

class ENetRegularizer_new
{
public:
    typedef Eigen::Map<Eigen::VectorXd> EigenMapVec;
    typedef Eigen::Map<Eigen::MatrixXd> EigenMapMat;
    typedef Eigen::VectorXd EigenVec;

private:
    static const int MIN_SIZE_TO_USE_MULTITHREAD = 100000;

private:

    static void gradcal(dense_sparse_matrix::DSMatrix& X, EigenVec& prob, EigenVec& beta, uint32_t& row_num, uint32_t& col_num,
              double lambda, double alpha, int threadNumber, ltt::allocator *pAlloc, EigenVec& grad);

    static void fvalue(EigenVec& Xb, EigenVec& Y, uint32_t row_num, uint32_t col_num, EigenVec& sol,
            double lambda, double alpha, double& fb);

    static double Qfun(EigenVec& alpha1, EigenVec& beta, double L, EigenVec& grad, uint32_t& col_num,
             double fb, double lambda, double alpha);

    static double Funcval(dense_sparse_matrix::DSMatrix& X, EigenVec& Y, EigenVec& alpha1,
                      double lambda, double alpha, uint32_t row_num, uint32_t col_num, int threadNumber, ltt::allocator *pAlloc);

    static void threshold(EigenVec& sol, double v, uint32_t col_num, EigenVec& alpha1);

    static void Toper(dense_sparse_matrix::DSMatrix& X, EigenVec& Y, EigenVec& sol, uint32_t row_num, uint32_t col_num,
             double L, double lambda, double alpha, int threadNumber, ltt::allocator *pAlloc, EigenVec& alpha1);

    static double gval(EigenVec& alpha1, double lambda, double alpha, uint32_t col_num);

public:
    static void lrPGDSolver(
	    dense_sparse_matrix::DSMatrix& x,
	    EigenVec& y,
	    double lambda,
	    double alpha,
	    int maxIter,
	    int threadNumber,
	    int& currentIter,
	    _STL::vector<double>& coefficient,
            double threshold,
	    ltt::allocator* pAlloc);
};

} // end of LOGR

}
#endif  //_ELASTICNETREG_H
