/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_gebd2_gebrd.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> gebrd_tuple;

// each matrix_size_range is a {m, lda}

// case when m = n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {20, 5},
    // normal (valid) samples
    {50, 50},
    {70, 100},
    {130, 130},
    {150, 200}};

const vector<int> n_size_range = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    16, 20, 120, 150};

const vector<vector<int>> large_matrix_size_range = {
    {152, 152},
    {640, 640},
    {1000, 1024},
};

const vector<int> large_n_size_range = {64, 98, 130, 220};

Arguments gebrd_setup_arguments(gebrd_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    int n_size = std::get<1>(tup);

    Arguments arg;

    arg.M = matrix_size[0];
    arg.N = n_size;
    arg.lda = matrix_size[1];

    arg.timing = 0;

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsp = min(arg.M, arg.N);
    arg.bsa = arg.lda * arg.N;

    return arg;
}

template <bool BLOCKED>
class GEBD2_GEBRD : public ::TestWithParam<gebrd_tuple>
{
protected:
    GEBD2_GEBRD() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void test_fixture()
    {
        Arguments arg = gebrd_setup_arguments(GetParam());

        if(arg.M == 0 && arg.N == 0)
            testing_gebd2_gebrd_bad_arg<BATCHED, STRIDED, BLOCKED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_gebd2_gebrd<BATCHED, STRIDED, BLOCKED, T>(arg);
    }
};

class GEBD2 : public GEBD2_GEBRD<false>
{
};

class GEBRD : public GEBD2_GEBRD<true>
{
};

// non-batch tests

TEST_P(GEBD2, __float)
{
    test_fixture<false, false, float>();
}

TEST_P(GEBD2, __double)
{
    test_fixture<false, false, double>();
}

TEST_P(GEBD2, __float_complex)
{
    test_fixture<false, false, rocblas_float_complex>();
}

TEST_P(GEBD2, __double_complex)
{
    test_fixture<false, false, rocblas_double_complex>();
}

TEST_P(GEBRD, __float)
{
    test_fixture<false, false, float>();
}

TEST_P(GEBRD, __double)
{
    test_fixture<false, false, double>();
}

TEST_P(GEBRD, __float_complex)
{
    test_fixture<false, false, rocblas_float_complex>();
}

TEST_P(GEBRD, __double_complex)
{
    test_fixture<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GEBD2, batched__float)
{
    test_fixture<true, true, float>();
}

TEST_P(GEBD2, batched__double)
{
    test_fixture<true, true, double>();
}

TEST_P(GEBD2, batched__float_complex)
{
    test_fixture<true, true, rocblas_float_complex>();
}

TEST_P(GEBD2, batched__double_complex)
{
    test_fixture<true, true, rocblas_double_complex>();
}

TEST_P(GEBRD, batched__float)
{
    test_fixture<true, true, float>();
}

TEST_P(GEBRD, batched__double)
{
    test_fixture<true, true, double>();
}

TEST_P(GEBRD, batched__float_complex)
{
    test_fixture<true, true, rocblas_float_complex>();
}

TEST_P(GEBRD, batched__double_complex)
{
    test_fixture<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(GEBD2, strided_batched__float)
{
    test_fixture<false, true, float>();
}

TEST_P(GEBD2, strided_batched__double)
{
    test_fixture<false, true, double>();
}

TEST_P(GEBD2, strided_batched__float_complex)
{
    test_fixture<false, true, rocblas_float_complex>();
}

TEST_P(GEBD2, strided_batched__double_complex)
{
    test_fixture<false, true, rocblas_double_complex>();
}

TEST_P(GEBRD, strided_batched__float)
{
    test_fixture<false, true, float>();
}

TEST_P(GEBRD, strided_batched__double)
{
    test_fixture<false, true, double>();
}

TEST_P(GEBRD, strided_batched__float_complex)
{
    test_fixture<false, true, rocblas_float_complex>();
}

TEST_P(GEBRD, strided_batched__double_complex)
{
    test_fixture<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEBD2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEBD2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEBRD,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEBRD,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));
