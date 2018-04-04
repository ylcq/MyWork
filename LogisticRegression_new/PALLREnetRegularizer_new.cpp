
#include "ltt/adp/iostream.hpp"
#include "ltt/adp/cmath.hpp"
#include "AFL/PAL/Classification/LogisticRegression_new/PALLREnetRegularizer_new.h"
#include "AFL/PAL/Classification/LogisticRegression_new/PALLogisticRegression_fix.h"
#include "AFL/PAL/palCancel.h"
#include "AFL/PAL/Utilities/Matrix/Matrix.hpp"
#include "AFL/PAL/Utilities/Matrix/EigenMatrixVectorOperation.h"

USE_TRACE(TRACE_PAL_LOGISTICREGRESSION);
USE_ERROR(AFL_PAL, ERR_PAL_CONVERGENCE_ERROR);
USE_TRACE(TRACE_PAL_TEST); //TRACE_ERROR(TRACE_PAL_TEST, "Error message");


namespace AFL_PAL
{
namespace LOGR
{

using ltt::vector;
using ltt::endl;

void ENetRegularizer_new::lrPGDSolver(
		dense_sparse_matrix::DSMatrix& data,
		EigenVec& Y,
		double lambda,
		double alpha,
		int maxIter,
		int threadNumber,
		int& currentIter,
		EigenVec& coefficient,
                double tol,
		ltt::allocator* pAlloc)
{

    uint32_t ncol = data.cols();
    uint32_t nrow = data.rows();

    Eigen::VectorXd alpha0(ncol);
    Eigen::VectorXd alpha1(ncol);
    Eigen::VectorXd beta(ncol);
    Eigen::VectorXd Xb(nrow);
    Eigen::VectorXd prob(nrow);
    Eigen::VectorXd grad(ncol);
    Eigen::VectorXd tmpVec(ncol);

    for(uint32_t i=0; i<ncol; ++i) {beta(i) = 0.0; alpha1(i) = 0.0; alpha0(i)= 0.0;} // initial

    int current_iter = 0;

    double t1 = 1.0;
    double t0 = 1.0;
    int iter = 0;
    double fb = 0.0;
    double L = 1.0;
    double etau = 1.1; double etad = 1.1; double delta = 0.8;
    int K = 2; int kre = 0;

    for(iter=1; iter<=maxIter; iter++)
    {
        bool isDec = false;

        current_iter++;

        dense_sparse_matrix::DSMatrix::matrixVectorProduct(data, beta, Xb, pAlloc, threadNumber);
        //matrixVectorProduct(data, beta, threadNumber, Xb, pAlloc);
        for(uint32_t i=0; i<nrow; i++)
        {
            prob(i) = -Y(i)/(1+exp(Y(i)*Xb(i)));
        }
        gradcal(data, prob, beta, nrow, ncol, lambda, alpha, threadNumber, pAlloc, grad);
        fvalue(Xb, Y, nrow, ncol, beta, lambda, alpha, fb);
        alpha0 = alpha1;
        tmpVec = beta - grad/L;
        threshold(tmpVec, lambda*alpha/L, ncol, alpha1);
        if(iter%10 == 1)
        {
            isDec = true;
            while (Funcval(data, Y, alpha1, lambda, alpha, nrow, ncol, threadNumber, pAlloc) >
                   Qfun(alpha1, beta, L, grad, ncol, fb, lambda, alpha))
            {
                L = etau * L;
                tmpVec = beta - grad/L;
                threshold(tmpVec, lambda*alpha/L, ncol, alpha1);
            }
        }
        Eigen::VectorXd tmpalphaVec(ncol);
        if(iter%100 == 1)
            Toper(data, Y, alpha1, nrow, ncol, L, lambda, alpha, threadNumber, pAlloc, tmpalphaVec);
        if((L*(alpha1-beta)).norm() < tol || ((iter%100)==1 && (L*(tmpalphaVec-alpha1)).norm() < tol))
            break;

        if(isDec)
            L = L/etad;

        t0 = t1;
        t1 = (1.0+sqrt(1+4*t0*t0))/2.0;
        beta = alpha1 + (t0-1.0)/t1*(alpha1 - alpha0);
        tmpVec = alpha1 - alpha0;
        if((iter > kre + K) &&
           (grad.dot(alpha1-alpha0)+ gval(alpha1, lambda, alpha, ncol)-gval(alpha0, lambda, alpha, ncol) > 0))
        {
            kre = iter;
            K *= 2;
            etad = delta*etad + 1-delta;
            t1 = 1;
            beta = alpha0;
            alpha1 = alpha0;
        }
    }

    /**
     * store the parameter of model into vector coefficients.
     */
    for(uint32_t i=0; i<ncol; i++)
    {
        coefficient(i) = alpha1(i);
    }

    currentIter = current_iter;

}

void ENetRegularizer_new::gradcal(dense_sparse_matrix::DSMatrix& X, EigenVec& prob, EigenVec& beta, uint32_t& row_num, uint32_t& col_num,
              double lambda, double alpha, int threadNumber, ltt::allocator* pAlloc, EigenVec& grad)
{
    dense_sparse_matrix::DSMatrix::matrixVectorProductATV(X, prob, grad, pAlloc, threadNumber);
    grad = grad/row_num;
    grad += lambda*(1-alpha)*beta;
    grad(0) -= lambda*(1-alpha)*beta(0);

}

void ENetRegularizer_new::fvalue(EigenVec& Xb, EigenVec& Y, uint32_t row_num, uint32_t col_num, EigenVec& sol,
            double lambda, double alpha, double& fb)
{
    fb = 0.0;
    for(uint32_t i=0; i<row_num; ++i)
    {
        double tmp = -Y(i) * Xb(i);
        if(tmp <= 0)
            fb += log(1+exp(tmp));
        else
            fb += tmp + log(1+exp(-tmp));
    }
    fb /= row_num;

    double norm = 0.0;
    for(uint32_t i=1; i<col_num; ++i)
    {
        norm += sol(i)*sol(i);
    }
    fb += 1.0/2*lambda*(1-alpha)*norm;

}

double ENetRegularizer_new::Qfun(Eigen::VectorXd& alpha1, Eigen::VectorXd& beta, double L, Eigen::VectorXd& grad, uint32_t& col_num,
             double fb, double lambda, double alpha)
{
    Eigen::VectorXd diff(col_num);
    diff = alpha1 - beta;
    double val = fb + grad.dot(diff) + L/2 * diff.dot(diff);
    for(uint32_t i=1; i<col_num;++i)
    {
        val += lambda*alpha*fabs(alpha1(i));
    }
    return val;
}

double ENetRegularizer_new::Funcval(dense_sparse_matrix::DSMatrix& X, EigenVec& Y, EigenVec& alpha1, 
                 double lambda, double alpha, uint32_t row_num, uint32_t col_num, int threadNumber, ltt::allocator *pAlloc)
{
    Eigen::VectorXd Xa(row_num);
    dense_sparse_matrix::DSMatrix::matrixVectorProduct(X, alpha1, Xa, pAlloc, threadNumber);
    double val = 0.0;
    fvalue(Xa, Y, row_num, col_num, alpha1, lambda, alpha, val);
    for(uint32_t i=1; i<col_num;++i)
    {
        val += lambda*alpha*fabs(alpha1(i));
    }
    return val;
}

void ENetRegularizer_new::threshold(Eigen::VectorXd& sol, double v, uint32_t col_num, Eigen::VectorXd& alpha1)
{
    for(uint32_t i=1; i<col_num; ++i)
    {
        if((sol(i) - v) > 0)
            alpha1(i) = sol(i) - v;
        else if((sol(i)+v) < 0)
            alpha1(i) = sol(i) + v;
        else
            alpha1(i) = 0;
    }
    alpha1(0) = sol(0);
}

void ENetRegularizer_new::Toper(dense_sparse_matrix::DSMatrix& X, EigenVec& Y, EigenVec& sol, uint32_t row_num, uint32_t col_num,
             double L, double lambda, double alpha, int threadNumber, ltt::allocator *pAlloc, EigenVec& alpha1)
{
    Eigen::VectorXd Xa(row_num);
    Eigen::VectorXd p(row_num);
    dense_sparse_matrix::DSMatrix::matrixVectorProduct(X, sol, Xa, pAlloc, threadNumber);
    for(uint32_t i=0; i<row_num; ++i)
    {
        p(i) = -Y(i)/(1+exp(Y(i)*Xa(i)));
    }
    Eigen::VectorXd tmpgrad(col_num);
    gradcal(X, p, sol, row_num, col_num, lambda, alpha, threadNumber, pAlloc, tmpgrad);
    tmpgrad = tmpgrad/L;
    tmpgrad = sol - tmpgrad;
    threshold(tmpgrad, alpha*lambda/L, col_num, alpha1);
}

double ENetRegularizer_new::gval(Eigen::VectorXd& alpha1, double lambda, double alpha, uint32_t col_num)
{
    double res =0.0;
    for(uint32_t i=1; i<col_num; ++i)
    {
        res += lambda*alpha*(fabs(alpha1(i)));
    }
    return res;
}

}//end of LOGR
}// end of AFL_PAL


