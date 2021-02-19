/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getri.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef vector<int> getri_tuple;

// each matrix_size_range vector is a {n, lda, singular}
// if singular = 1, then the used matrix for the tests is singular

// case when n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {20, 5, 0},
    // normal (valid) samples
    {32, 32, 0},
    {50, 50, 1},
    {70, 100, 0},
    {100, 150, 1}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{192, 192, 1}, {500, 600, 1}, {640, 640, 0}, {1000, 1024, 0}, {1200, 1230, 0}};

Arguments getri_setup_arguments(getri_tuple tup)
{
    // vector<int> matrix_size = std::get<0>(tup);

    Arguments arg;

    arg.N = tup[0];
    arg.lda = tup[1];

    arg.timing = 0;
    arg.singular = tup[2];

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsp = arg.N;
    arg.bsa = arg.lda * arg.N;

    return arg;
}

class GETRI : public ::TestWithParam<getri_tuple>
{
protected:
    GETRI() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void test_fixture()
    {
        Arguments arg = getri_setup_arguments(GetParam());

        if(arg.N == 0)
            testing_getri_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_getri<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_getri<BATCHED, STRIDED, T>(arg);
    }
};

// non-batch tests

TEST_P(GETRI, __float)
{
    test_fixture<false, false, float>();
}

TEST_P(GETRI, __double)
{
    test_fixture<false, false, double>();
}

TEST_P(GETRI, __float_complex)
{
    test_fixture<false, false, rocblas_float_complex>();
}

TEST_P(GETRI, __double_complex)
{
    test_fixture<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GETRI, batched__float)
{
    test_fixture<true, true, float>();
}

TEST_P(GETRI, batched__double)
{
    test_fixture<true, true, double>();
}

TEST_P(GETRI, batched__float_complex)
{
    test_fixture<true, true, rocblas_float_complex>();
}

TEST_P(GETRI, batched__double_complex)
{
    test_fixture<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GETRI, strided_batched__float)
{
    test_fixture<false, true, float>();
}

TEST_P(GETRI, strided_batched__double)
{
    test_fixture<false, true, double>();
}

TEST_P(GETRI, strided_batched__float_complex)
{
    test_fixture<false, true, rocblas_float_complex>();
}

TEST_P(GETRI, strided_batched__double_complex)
{
    test_fixture<false, true, rocblas_double_complex>();
}

// outofplace_batched tests

TEST_P(GETRI, outofplace_batched__float)
{
    test_fixture<true, false, float>();
}

TEST_P(GETRI, outofplace_batched__double)
{
    test_fixture<true, false, double>();
}

TEST_P(GETRI, outofplace_batched__float_complex)
{
    test_fixture<true, false, rocblas_float_complex>();
}

TEST_P(GETRI, outofplace_batched__double_complex)
{
    test_fixture<true, false, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, GETRI, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GETRI, ValuesIn(matrix_size_range));
