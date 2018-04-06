#ifndef __PALLOGRMODEL_H
#define __PALLOGRMODEL_H

#include <ltt/adp/vector.hpp>
#include <TRexUtils/Parallel.h>
#include <TRexUtils/ParallelDispatcher.h>
#include "AFL/PAL/pal.h"
#include "AFL/PAL/Classification/LogisticRegression_new/PALLogisticRegression_fix.h"
#include "AFL/PAL/Classification/LogisticRegression_new/PALLREnetRegularizer_new.h"
#include "AFL/PAL/Classification/LogisticRegression_new/ObjectFunction.h"
#include "AFL/PAL/PALTableTypeVerifier.h"
#include "AFL/PAL/Statistics/CommonFunctions/PValueCalc.hpp"
#include "AFL/PAL/Utilities/Matrix/Matrix.hpp"
#include "AFL/PAL/Utilities/Matrix/DenseSparseMatrix.hpp"
#include "AFL/PAL/Utilities/Optimization/Solver.hpp"
#include "AFL/PAL/Utilities/PMML/palGenPmml.h"
#include "AFL/PAL/Utilities/PMML/regressionModel.h"
#include "AFL/PAL/palCancel.h"
#include "AFL/PAL/palCommonFunction.hpp"
#include "AFL/PAL/palParameters.h"
#include "AFL/PAL/palTableTypes.h"

namespace AFL_PAL
{
namespace LOGR
{
class LogRModel
{
public:
    explicit LogRModel(ltt::allocator& alloc):
        m_alloc(&alloc),
        m_coefficient(alloc),
        m_categoryEachColumn(alloc),
        m_continuousColNum(0),
        m_categoryColNum(0),
        m_retBackCoeff(alloc),
        m_nretBackCoeff(alloc),
        ia(alloc),
        mm(alloc),
        finalIter(0),
        solutionStatus(0),
        minusMLE(0.0),
        expandColSize(0),
        feature_dimension(0)
    {  }

public:
    void modelTrain(dense_sparse_matrix::DSMatrix& dsMatrix, Eigen::VectorXd& yValue,
                    LR_ARGUMENTS_NEW& LR_Args, TRexCommonObjects::InternalTable* argsTab,
                    double lambda, double alpha, size_t threadNumber);

    _STL::vector<double>& getCoefficient() {return m_coefficient;}

    _STL::vector<double>& getRetBackCoefficient() {return m_retBackCoeff;}

    _STL::vector<double>& getNoRetBackCoefficient() {return m_nretBackCoeff;}

    ltt::vector<int>& getIA() {return ia;}

    ltt::vector<int>& getMM() {return mm;}

    size_t getFinalIter() {return finalIter;}

    size_t getSolutionStatus() {return solutionStatus;}

    double getMinusMLE() {return minusMLE;}

    size_t getExpandColSize() {return expandColSize;}

    size_t getFeatureDimension() {return feature_dimension;}

    void setCategoryEachColumn(_STL::vector<_STL::vector<_STL::string>>& categoryEachColumn){
        m_categoryEachColumn = categoryEachColumn;
    }

    _STL::vector<_STL::vector<_STL::string>>& getCategoryEachColumn() {return m_categoryEachColumn;}

    void setContinuousColNum(uint32_t continuousColNum) {m_continuousColNum = continuousColNum;}
    void setCategoryColNum(uint32_t categoryColNum) {m_categoryColNum = categoryColNum;}
    uint32_t getContinuousColNum() {return m_continuousColNum;}
    uint32_t getCategoryColNum() {return m_categoryColNum;}


private:
    ltt::allocator* m_alloc;
    _STL::vector<double> m_coefficient;
    _STL::vector<_STL::vector<_STL::string>> m_categoryEachColumn;
    uint32_t m_continuousColNum;
    uint32_t m_categoryColNum;
    _STL::vector<double> m_retBackCoeff;
    _STL::vector<double> m_nretBackCoeff;
    ltt::vector<int> ia;
    ltt::vector<int> mm;
    int finalIter;
    size_t solutionStatus;
    double minusMLE = 0;
    size_t expandColSize;
    size_t feature_dimension;

};


}// end of LOGR
}// end of AFL_PAL

#endif
