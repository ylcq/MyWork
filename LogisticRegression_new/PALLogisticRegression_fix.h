/*
 * PALLogisticRegression_fix.h
 *
 *  Created on: April 4, 2018
 *      Author: i323463
 */

#ifndef PALLOGISTICREGRESSION_NEW_H_
#define PALLOGISTICREGRESSION_NEW_H_
#include "AFL/PAL/pal.h"
#include "TRexUtils/ParallelDispatcher.h"
#include "TRexCommonObjects/InternalTable/InternalTable.h"
#include "AFL/PAL/Utilities/ProcessBase/FeatureSelector.h"
#include "AFL/PAL/Utilities/ProcessBase/FeatureSelectorForecast.h"
#include "AFL/PAL/Utilities/ProcessBase/ProcessBase.h"
#include "AFL/PAL/APIExport.hpp"
#include "AFL/PAL/Utilities/Matrix/Matrix.hpp"
#include "AFL/PAL/Utilities/Matrix/DenseSparseMatrix.hpp"
#include "AFL/PAL/Utilities/DataTableView/DataTableView.h"
#include "AFL/PAL/Utilities/CrossValidation/ParameterGroup.h"
#include "AFL/PAL/Utilities/CrossValidation/ParameterSelector.h"
#include "AFL/PAL/Utilities/CrossValidation/ParameterSearch.h"
#include "AFL/PAL/Utilities/CrossValidation/Help.h"
#include "AFL/PackageManager/PackageManager/AFLPM_PackageManager.hpp"
#include "AFL/PAL/Classification/LogisticRegression_new/LogRHelper.h"

namespace AFL_PAL
{
namespace LOGR{

class  PALLogisticRegression_new : public ProcessBase
{
    typedef ltt::smartptr_handle<TRexCommonObjects::InternalTable> smptr_itab;

public:
    PALLogisticRegression_new(
            TRexCommonObjects::InternalTable *dataTab,
            TRexCommonObjects::InternalTable *iargsTab,
            TRexCommonObjects::InternalTable *coefficientTab,
            TRexCommonObjects::InternalTable *pmmlTab,
            TRexCommonObjects::InternalTable *statisticTab,
            TRexCommonObjects::InternalTable *parameterTab,
            AFLProgressIndicator* prgIndicator=nullptr):
        dataTab(dataTab),
        iargsTab(iargsTab),
        coefficientTab(coefficientTab),
        statisticsTab(statisticTab),
        pmmlTab(pmmlTab),
        hAlloc(getGlobalPALAllocator().createSubAllocator("PALLogisticRegressionAllocator")),
        m_OutputStat(true),
        LR_Args(*hAlloc),
        m_parameterTab(parameterTab),
        coefficientKey(*hAlloc),
        m_prgIndicator(prgIndicator),
        m_isTimeOut(false),
        m_hasRsl(false)
    {
    }

    PALLogisticRegression_new(
            const smptr_itab     &dataTab,
            const smptr_itab     &iargsTab,
                  smptr_itab     &coefficientTab,
                  smptr_itab     &pmmlTab,
                  smptr_itab     &statisticTab,
                  smptr_itab     &parameterTab,
                  AFLProgressIndicator* prgIndicator=nullptr):
        dataTab(dataTab.get()),
        iargsTab(iargsTab.get()),
        coefficientTab(coefficientTab.get()),
        statisticsTab(statisticTab.get()),
        pmmlTab(pmmlTab.get()),
        hAlloc(getGlobalPALAllocator().createSubAllocator("PALLogisticRegressionAllocator")),
        m_OutputStat(true),
        LR_Args(*hAlloc),
        m_parameterTab(parameterTab.get()),
        coefficientKey(*hAlloc),
        m_prgIndicator(prgIndicator),
        m_isTimeOut(false),
        m_hasRsl(false)
    {
    }

    void execute();

    // called by feature selector
    TRexCommonObjects::InternalTable* getInputTable(int index)
    {
        return dataTab;
    }

    // called by feature selector
    TRexCommonObjects::InternalTable* getControlTable()
    {
        return iargsTab;
    }

    // called by feature selector
    TRexCommonObjects::InternalTable* getOutputTable(int index)
    {
        if(index == 0)
        {
            return coefficientTab;
        }
        else
        {
            return pmmlTab;
        }
    }

    void composeDataTable(
            TRexCommonObjects::InternalTable* featureTable,
            TRexCommonObjects::ColumnBase* target);

    _STL::string getDefaultTargetName()
    {
        int numColumns = dataTab->getNumColumns();
        return dataTab->getColumn(numColumns - 1)->getName();
    }

protected:
    void setupPreProcessChain()
    {
        //add2Chain(&fs);
        m_feature_select.reset(
                new FeatureSelector<PALLogisticRegression_new>(this, *hAlloc));
        add2Chain(m_feature_select.get());
    }

private:
    /*!
     * @brief PALLOGISTICREGRESSION POP.
     *
     * @note In case of error, a AFL_PAL exception is thrown.
     *
     * @param dataTab For the regression data.
     * @param iargsTab of parameter.
     * @param coefficientTab of coefficient.
     * @param fittedValueTab  the value of Y.
     * @will throw error if there is error.
     */
    void getArguments();

    void checkAndValidateTables();

    void getCategoryVec(ltt_adp::vector<bool> &isCategoryVec, ltt_adp::vector<bool>& categoryColVec);

    void getCoefficientKey(ltt::vector<_STL::string>& key,
            ltt_adp::vector<bool> isCategoryVec);

    void loadLabel(Eigen::VectorXd& label);

    void retBackCoefficient(Eigen::VectorXd& orgCoeff, size_t& contNum);

    void sortCoefficient(Eigen::VectorXd& orgCoeff, ltt_adp::vector<uint32_t>& cateNumofEachColumn,
               ltt_adp::vector<uint32_t>&insPos, size_t contNum, ltt::vector<double>& backCoeff, uint32_t& expandColSize);

    void writeCoefficientToTable(ltt::vector<double>& coefficient, ltt::vector<uint32_t>& insPos,
                         uint32_t& expandColSize, ltt::vector<double>& pvalue, ltt::vector<double>& zscore);

    void outputStatistic(double& minusMLE, uint32_t& feature_dimension, double& objVal,
                         int32_t& finalIter, int& solutionStatus);

    void calculateObjAndMLE(dense_sparse_matrix::DSMatrix& dsMat, Eigen::VectorXd& yValue,
                            Eigen::VectorXd& coeff, uint32_t& expandColSize,
                            uint32_t&feature_dimension, double& objValue, double& minusMLE);

    void calculatePValue(dense_sparse_matrix::DSMatrix& dsMat, Eigen::VectorXd& coeff_NewValue,
                         ltt_adp::vector<uint32_t>& insPos, ltt::vector<double>& zscore, ltt::vector<double>& pvalue);

    template<typename T>
    void outputVector(ltt_adp::vector<T> vec);

    TRexCommonObjects::InternalTable *dataTab;
    TRexCommonObjects::InternalTable *iargsTab;
    TRexCommonObjects::InternalTable *coefficientTab;
    TRexCommonObjects::InternalTable *statisticsTab;
    TRexCommonObjects::InternalTable *pmmlTab;
    ltt::allocator_handle hAlloc;
    _STL::unique_ptr<FeatureSelector<PALLogisticRegression_new>> m_feature_select;
    ltt::smartptr_handle<InternalTable> mArgsTabPtr;
    ltt::smartptr_handle<InternalTable> mDataTabPtr;
    ltt_adp::set<uint32_t> categoryColSet;
    bool outputCoeByKey;
    bool m_OutputStat;
    LR_ARGUMENTS_NEW LR_Args;

    TRexCommonObjects::InternalTable *m_parameterTab;
    ltt::vector<_STL::string> coefficientKey;

    AFLProgressIndicator* m_prgIndicator;
    bool m_isTimeOut;
    bool m_hasRsl;

};
} // end of LOGR
}  // end of AFL_PAL
#endif  // PALLOGISTICREGRESSION_H
