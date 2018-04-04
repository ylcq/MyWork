#ifndef __PALLOGRDATASTRUCT_H
#define __PALLOGRDATASTRUCT_H

#include "AFL/PAL/Utilities/DataTableView/DataTableView.h"
#include <ltt/adp/vector.hpp>
#include <ltt/vector.hpp>
#include "AFL/PAL/pal.h"
#include <TRexUtils/ParallelDispatcher.h>
#include <TRexCommonObjects/InternalTable/InternalTable.h>
#include "AFL/PAL/Utilities/Matrix/DenseSparseMatrix.hpp"

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
        yValue(),
        dsMatrix(alloc),
        meanVec(alloc),
        stdVec(alloc),
        categoryVec(alloc),
        insPos(alloc),
        categoryNumOfEachColumn(alloc),
        m_alloc(&alloc)
    {  }

public:
    void loadData(Utility::DataTableViewBase const* pDataView,
                  ltt::vector<_STL::string>& classMap,
                  ltt_adp::vector<bool>& isCategoryVec, int& method, bool& isCV);

public:
    dense_sparse_matrix::DSMatrix& getDSMatrix() {return dsMatrix;}
    Eigen::VectorXd& getYValue() {return yValue;}
    ltt::vector<double>& getMeanVec() {return meanVec;}
    ltt::vector<double>& getStdVec() {return stdVec;}
    ltt::vector<ltt::vector<_STL::string>>& getCategoryVec() {return categoryVec;}


private:
    void loadFeatureData(Utility::DataTableViewBase const* dataView,
                         ltt_adp::vector<bool>& isCategoryVec, bool& isCV);

    void loadLabelData(Utility::DataTableViewBase const* dataView,
                       ltt::vector<_STL::string>& classMap, int& method);

    void handleCategoryColumn(Utility::DataColumnView const* categoryCol, uint32_t& nrow,
                       _STL::vector<T>& tripletList, uint32_t& cateInd, uint32_t& dictSize);

private:
    Eigen::VectorXd yValue;
    dense_sparse_matrix::DSMatrix dsMatrix;
    ltt::vector<double> meanVec;
    ltt::vector<double> stdVec;
    ltt::vector<ltt::vector<_STL::string>> categoryVec;
    ltt_adp::vector<uint32_t> insPos;
    ltt_adp::vector<uint32_t> categoryNumOfEachColumn;
    ltt::allocator* m_alloc;
};

class LogRPredictData
{
public:
    explicit LogRPredictData(ltt::allocator& alloc):
        m_alloc(&alloc),
        predictData(),
        categoryVec(alloc)
    {  }

public:
    void loadData(Utility::DataTableViewBase const* pDataView,
                  ltt_adp::vector<bool>& isCategoryVec);

    Eigen::MatrixXd& getPredictData() {return predictData;}

    ltt::vector<ltt::vector<_STL::string>>& getCategoryVec() {return categoryVec;}

private:
    void loadFeatureData(Utility::DataTableViewBase const* dataView, ltt_adp::vector<bool>& isCategoryVec);

private:
    ltt::allocator* m_alloc;
    Eigen::MatrixXd predictData;
    ltt::vector<ltt::vector<_STL::string>> categoryVec;
};


class LogRCompareData
{
public:
    explicit LogRCompareData(ltt::allocator& alloc):
        m_alloc(&alloc),
        compareData()
    {  }

public:
   void loadData(Utility::DataTableViewBase const* pDataView,
                 ltt::vector<_STL::string>& classMap);

   Eigen::VectorXd& getCompareData() {return compareData;}

   _STL::vector<_STL::string>& getStrLabels() {return stringLabels;}

private:
    ltt::allocator* m_alloc;
    Eigen::VectorXd compareData;
    _STL::vector<_STL::string> stringLabels;
};

}// end of LOGR
}// end of AFL_PAL

#endif








