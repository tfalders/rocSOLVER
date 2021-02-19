/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_potf2_potrf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, char> potrf_tuple;

// each size_range vector is a {N, lda, singular}
// if singular = 1, then the used matrix for the tests is not positive definite

// each uplo_range is a {uplo}

// case when n = 0 and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<char> uplo_range = {'L', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {10, 2, 0},
    // normal (valid) samples
    {10, 10, 1},
    {20, 30, 0},
    {50, 50, 1},
    {70, 80, 0}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 0}, {640, 960, 1}, {1000, 1000, 0}, {1024, 1024, 1}, {2000, 2000, 0},
};

Arguments potrf_setup_arguments(potrf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    char uplo = std::get<1>(tup);

    Arguments arg;

    arg.N = matrix_size[0];
    arg.lda = matrix_size[1];

    arg.uplo_option = uplo;

    arg.timing = 0;
    arg.singular = matrix_size[2];

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsa = arg.lda * arg.N;

    return arg;
}

template <bool BLOCKED>
class POTF2_POTRF : public ::TestWithParam<potrf_tuple>
{
protected:
    POTF2_POTRF() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void test_fixture()
    {
        Arguments arg = potrf_setup_arguments(GetParam());

        if(arg.uplo_option == 'L' && arg.N == 0)
            testing_potf2_potrf_bad_arg<BATCHED, STRIDED, BLOCKED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_potf2_potrf<BATCHED, STRIDED, BLOCKED, T>(arg);

        arg.singular = 0;
        testing_potf2_potrf<BATCHED, STRIDED, BLOCKED, T>(arg);
    }
};

class POTF2 : public POTF2_POTRF<false>
{
};

class POTRF : public POTF2_POTRF<true>
{
};

// non-batch tests

TEST_P(POTF2, __float)
{
    test_fixture<false, false, float>();
}

TEST_P(POTF2, __double)
{
    test_fixture<false, false, double>();
}

TEST_P(POTF2, __float_complex)
{
    test_fixture<false, false, rocblas_float_complex>();
}

TEST_P(POTF2, __double_complex)
{
    test_fixture<false, false, rocblas_double_complex>();
}

TEST_P(POTRF, __float)
{
    test_fixture<false, false, float>();
}

TEST_P(POTRF, __double)
{
    test_fixture<false, false, double>();
}

TEST_P(POTRF, __float_complex)
{
    test_fixture<false, false, rocblas_float_complex>();
}

TEST_P(POTRF, __double_complex)
{
    test_fixture<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(POTF2, batched__float)
{
    test_fixture<true, true, float>();
}

TEST_P(POTF2, batched__double)
{
    test_fixture<true, true, double>();
}

TEST_P(POTF2, batched__float_complex)
{
    test_fixture<true, true, rocblas_float_complex>();
}

TEST_P(POTF2, batched__double_complex)
{
    test_fixture<true, true, rocblas_double_complex>();
}

TEST_P(POTRF, batched__float)
{
    test_fixture<true, true, float>();
}

TEST_P(POTRF, batched__double)
{
    test_fixture<true, true, double>();
}

TEST_P(POTRF, batched__float_complex)
{
    test_fixture<true, true, rocblas_float_complex>();
}

TEST_P(POTRF, batched__double_complex)
{
    test_fixture<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(POTF2, strided_batched__float)
{
    test_fixture<false, true, float>();
}

TEST_P(POTF2, strided_batched__double)
{
    test_fixture<false, true, double>();
}

TEST_P(POTF2, strided_batched__float_complex)
{
    test_fixture<false, true, rocblas_float_complex>();
}

TEST_P(POTF2, strided_batched__double_complex)
{
    test_fixture<false, true, rocblas_double_complex>();
}

TEST_P(POTRF, strided_batched__float)
{
    test_fixture<false, true, float>();
}

TEST_P(POTRF, strided_batched__double)
{
    test_fixture<false, true, double>();
}

TEST_P(POTRF, strided_batched__float_complex)
{
    test_fixture<false, true, rocblas_float_complex>();
}

TEST_P(POTRF, strided_batched__double_complex)
{
    test_fixture<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POTF2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTF2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POTRF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTRF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));
