#ifndef __PALLOGRDATASTRUCT_H
#define __PALLOGRDATASTRUCT_H

#include "AFL/PAL/Utilities/DataTableView/DataTableView.h"
#include <ltt/adp/vector.hpp>
#include <ltt/vector.hpp>
#include "AFL/PAL/pal.h"
#include <TRexUtils/ParallelDispatcher.h>
#include <TRexCommonObjects/InternalTable/InternalTable.h>
#include "AFL/PAL/Utilities/Matrix/DenseSparseMatrix.hpp"
#include "AFL/PAL/Classification/LogisticRegression_new/LogRHelper.h"

namespace AFL_PAL
{
namespace LOGR
{
class LogRTrainData
{
public:
    typedef Eigen::Triplet<double> T;

public:
    explicit LogRTrainData(ltt::allocator& alloc):
        m_yValue(),
        m_dsMatrix(alloc),
        m_meanVec(alloc),
        m_stdVec(alloc),
        m_categoryEachColumn(alloc),
        m_insPos(alloc),
        m_categoryNumOfEachColumn(alloc),
        m_hasCateThanOneIdx(alloc),
        m_alloc(&alloc),
        m_continuousColNum(0),
        m_categoryColNum(0)
    {  }

public:
    void loadData(Utility::DataTableViewBase const* pDataView,
                  ltt::vector<_STL::string>& classMap,
                  ltt_adp::vector<bool>& isCategoryVec, int& method, bool& isCV);

public:
    void standardize();
    dense_sparse_matrix::DSMatrix& getDSMatrix() {return m_dsMatrix;}
    Eigen::VectorXd& getYValue() {return m_yValue;}
    ltt_adp::vector<double>& getMeanVec() {return m_meanVec;}
    ltt_adp::vector<double>& getStdVec() {return m_stdVec;}
    _STL::vector<_STL::vector<_STL::string>>& getCategoryEachColumn() {return m_categoryEachColumn;}
    ltt_adp::vector<uint32_t>& getInsPos() {return m_insPos;}
    ltt_adp::vector<uint32_t>& getCategoryNumOfEachColumn() {return m_categoryNumOfEachColumn;}
    uint32_t getContinuousColNum() {return m_continuousColNum;}
    uint32_t getCategoryColNum() {return m_categoryColNum;}
    ltt_adp::vector<uint32_t>& getHasCateThanOneIdx() {return m_hasCateThanOneIdx;}

private:
    void loadFeatureData(Utility::DataTableViewBase const* dataView,
                         ltt_adp::vector<bool>& isCategoryVec, bool& isCV);

    void loadLabelData(Utility::DataTableViewBase const* dataView,
                       ltt::vector<_STL::string>& classMap, int& method);

    void handleCategoryColumn(Utility::DataColumnView const* categoryCol, uint32_t& nrow,
                       _STL::vector<T>& tripletList, uint32_t& cateInd, uint32_t& dictSize);

private:
    Eigen::VectorXd m_yValue;
    dense_sparse_matrix::DSMatrix m_dsMatrix;
    ltt_adp::vector<double> m_meanVec;
    ltt_adp::vector<double> m_stdVec;
    _STL::vector<_STL::vector<_STL::string>> m_categoryEachColumn;
    ltt_adp::vector<uint32_t> m_insPos;
    ltt_adp::vector<uint32_t> m_categoryNumOfEachColumn;
    ltt_adp::vector<uint32_t> m_hasCateThanOneIdx;
    ltt::allocator* m_alloc;
    uint32_t m_continuousColNum;
    uint32_t m_categoryColNum;
};

class LogRPredictData
{
public:
    typedef Eigen::Triplet<double> T;

public:
    explicit LogRPredictData(ltt::allocator& alloc):
        m_alloc(&alloc),
        m_dsMatrix(alloc),
        categoryVec(alloc),
        eachColumnCategoryIdx(alloc),
        m_continuousColNum(0),
        m_categoryColNum(0)
    {  }

public:
    void loadData(Utility::DataTableViewBase const* pDataView,
                  ltt_adp::vector<bool>& isCategoryVec);

    dense_sparse_matrix::DSMatrix& getPredictData() {return m_dsMatrix;}

    ltt_adp::vector<ltt_adp::vector<_STL::string>>& getCategoryVec() {return categoryVec;}

    uint32_t getContinuousColNum() {return m_continuousColNum;}

    uint32_t getCategoryColNum() {return m_categoryColNum;}

    ltt_adp::vector<ltt::map<_STL::string, ltt_adp::vector<uint32_t>>> getEachColumnCategoryIdx() {return eachColumnCategoryIdx;}

private:
    void loadFeatureData(Utility::DataTableViewBase const* dataView, ltt_adp::vector<bool>& isCategoryVec);

private:
    ltt::allocator* m_alloc;
    dense_sparse_matrix::DSMatrix m_dsMatrix;
    //Eigen::MatrixXd predictData;
    ltt_adp::vector<ltt_adp::vector<_STL::string>> categoryVec;
    ltt_adp::vector<ltt::map<_STL::string, ltt_adp::vector<uint32_t>>> eachColumnCategoryIdx;
    uint32_t m_continuousColNum;
    uint32_t m_categoryColNum;
};


class LogRCompareData
{
public:
    explicit LogRCompareData(ltt::allocator& alloc):
        m_alloc(&alloc),
        compareData(),
        m_stringLabels(alloc)
    {  }

public:
   void loadData(Utility::DataTableViewBase const* pDataView,
                 ltt::vector<_STL::string>& classMap);

   Eigen::VectorXd& getCompareData() {return compareData;}

   _STL::vector<_STL::string>& getStrLabels() {return m_stringLabels;}

private:
    ltt::allocator* m_alloc;
    Eigen::VectorXd compareData;
    _STL::vector<_STL::string> m_stringLabels;
};

}// end of LOGR
}// end of AFL_PAL

#endif








