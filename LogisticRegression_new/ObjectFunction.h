/*
 * ObjectFunction.h
 *
 *  Created on: Aril 6, 2018
 *      Author: root
 */

#ifndef _PAL_LOGR_OBJECTFUNCTION_H
#define _PAL_LOGR_OBJECTFUNCTION_H

#include "AFL/PAL/Utilities/Optimization/VectorCalculator.hpp"
#include "AFL/PAL/Utilities/Optimization/State.hpp"
#include "AFL/PAL/Utilities/Matrix/Matrix.hpp"
#include "AFL/PAL/Utilities/Matrix/DenseSparseMatrix.hpp"

namespace AFL_PAL
{
namespace LOGR
{
class ObjectFunction
{
       typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> EigenMatCol;                          
       typedef Eigen::Map<EigenMatCol> EigenMapMatCol;
       typedef Eigen::Map<Eigen::VectorXd> EigenMapVec;
       typedef Eigen::VectorXd EigenVec;
       typedef Eigen::MatrixXd EigenMat;
       typedef Eigen::Map<Eigen::ArrayXd> EigenMapArr;

public:
    ObjectFunction(
            dense_sparse_matrix::DSMatrix& tmpData,
            EigenVec& categ,
            const uint32_t tmpdim,
            const int numthread,
            ltt::allocator& alloc,
            double c = 0):
        m_vecData(tmpData),
        m_vecCat(categ),
        m_scale(tmpdim, alloc),
        m_Dimension(tmpdim),
        m_ttl(m_vecData.rows()),
        m_numthread(numthread),
        m_thetax(m_ttl, alloc),
        m_p(m_ttl, alloc),
        m_ep(m_ttl, alloc),
        m_py(m_ttl, alloc),
        m_Alloc(&alloc),
        m_c(c)
    {

    }

    virtual ~ObjectFunction()
    {
    }

    // obj = logistic likelihood
    int updateObjective(struct State& out_state);
    int updateGradient(struct State& out_state);
    int updateState(struct State& out_state);

    void normalization();

    void retBack(ltt::vector<double>& tar);

    void calculateLogExpByParallel(const EigenMapArr& thetaxArr, const EigenMapArr& catArr, size_t threadNumber,
                                   ltt::allocator* pAlloc, double& sum, EigenMapArr& pArr);

protected:
    void setXdata();
    dense_sparse_matrix::DSMatrix& m_vecData;
    EigenVec& m_vecCat;
    ltt::vector<double> m_scale;
    uint32_t            m_Dimension;
    size_t              m_ttl;
    size_t              m_numthread;
    ltt::vector<double> m_thetax;
    ltt::vector<double> m_p;
    ltt::vector<double> m_ep;
    ltt::vector<double> m_py;
    ltt::allocator* m_Alloc;
    double m_c;

};


class calculateSumJob : public TRexUtils::Parallel::JobBase
{
    typedef Eigen::Map<Eigen::ArrayXd> EigenMapArr;
public:
    calculateSumJob(TRexUtils::Parallel::SimpleBlockPartitioner<size_t>& partitioner,
                    const EigenMapArr& thetax, const EigenMapArr& vecCat,
                    double& sum,  EigenMapArr& p):
        m_partitioner(&partitioner), m_thetax(thetax), m_vecCat(vecCat), m_sum(sum), m_p(p) {}

    virtual void getMethod(ltt::ostream &os) const { os << "calculateSumJob"; }
    virtual void getDetails(ltt::ostream &os) const { os << "calculateSumJob"; }
    TRexUtils::Parallel::JobResult run()
    {
           size_t m_start=0, m_end=0;
           while(m_partitioner->getNextBlock(m_start, m_end)){
               Eigen::ArrayXd logexpArr(m_end-m_start);
               m_p.segment(m_start, m_end-m_start) = exp(-m_thetax.segment(m_start, m_end-m_start).abs());

               logexpArr = (m_thetax.segment(m_start, m_end-m_start)+m_thetax.segment(m_start, m_end-m_start).abs())/2 + log1p(m_p.segment(m_start, m_end-m_start)) - m_vecCat.segment(m_start, m_end-m_start) * m_thetax.segment(m_start, m_end-m_start);

               m_p.segment(m_start, m_end-m_start) = (1 + m_p.segment(m_start, m_end-m_start)).inverse();
               for(uint32_t i=m_start; i != m_end; ++i){
                   if(m_thetax(i) < 0){
                       m_p(i) = 1.0 - m_p(i);
                   }
               }
               m_sum += logexpArr.matrix().sum();
           }

           return TRexUtils::Parallel::Done;
        }

    private:
        TRexUtils::Parallel::SimpleBlockPartitioner<size_t> * m_partitioner;
        const EigenMapArr& m_thetax;
        const EigenMapArr& m_vecCat;
        double& m_sum;
        EigenMapArr& m_p;


    };
}// end of LOGR
}// end of AFL_PAL
#endif  // OBJECTFUNCTION_H
