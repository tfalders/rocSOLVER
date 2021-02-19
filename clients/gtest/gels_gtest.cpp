/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_gels.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<int, int, int, int, int> gels_params_A;
typedef std::tuple<int, rocsolver_op_char> gels_params_B;

typedef std::tuple<gels_params_A, gels_params_B> gels_tuple;

// each A_range tuple is a {M, N, lda, ldb, singular};
// if singular = 1, then the used matrix for the tests is singular

// each B_range tuple is a {nrhs, trans};

// case when N = nrhs = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<gels_params_A> matrix_sizeA_range = {
    // quick return
    {0, 0, 0, 0, 0},
    // invalid
    {-1, 1, 1, 1, 0},
    {1, -1, 1, 1, 0},
    {10, 10, 10, 1, 0},
    {10, 10, 1, 10, 0},
    // not yet implemented
    {10, 1, 10, 10, 0},
    // normal (valid) samples
    {20, 20, 20, 20, 1},
    {30, 20, 40, 30, 0},
    {40, 20, 40, 40, 1},
};
const vector<gels_params_B> matrix_sizeB_range = {
    // quick return
    {0, 'N'},
    // invalid
    {-1, 'N'},
    // not yet implemented
    {1, 'T'},
    {1, 'C'},
    // normal (valid) samples
    {10, 'N'},
    {20, 'N'},
    {30, 'N'},
};

// for daily_lapack tests
const vector<gels_params_A> large_matrix_sizeA_range = {
    {75, 25, 75, 75, 0},
    {150, 150, 150, 150, 1},
    {500, 50, 600, 600, 0},
    {1000, 500, 1000, 1000, 1},
};
const vector<gels_params_B> large_matrix_sizeB_range = {
    {100, 'N'},
    {200, 'N'},
    {500, 'N'},
    {1000, 'N'},
};

Arguments gels_setup_arguments(gels_tuple tup)
{
    gels_params_A matrix_sizeA = std::get<0>(tup);
    gels_params_B matrix_sizeB = std::get<1>(tup);

    Arguments arg;

    arg.M = std::get<0>(matrix_sizeA);
    arg.N = std::get<1>(matrix_sizeA);
    arg.lda = std::get<2>(matrix_sizeA);
    arg.ldb = std::get<3>(matrix_sizeA);
    arg.singular = std::get<4>(matrix_sizeA);

    arg.K = std::get<0>(matrix_sizeB);
    arg.transA_option = std::get<1>(matrix_sizeB);

    arg.timing = 0;

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsa = arg.lda * arg.N;
    arg.bsb = arg.ldb * arg.K;

    return arg;
}

class GELS : public ::TestWithParam<gels_tuple>
{
protected:
    GELS() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void test_fixture()
    {
        Arguments arg = gels_setup_arguments(GetParam());

        if(arg.M == 0 && arg.K == 0)
            testing_gels_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_gels<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_gels<BATCHED, STRIDED, T>(arg);
    }
};

// non-batch tests

TEST_P(GELS, __float)
{
    test_fixture<false, false, float>();
}

TEST_P(GELS, __double)
{
    test_fixture<false, false, double>();
}

TEST_P(GELS, __float_complex)
{
    test_fixture<false, false, rocblas_float_complex>();
}

TEST_P(GELS, __double_complex)
{
    test_fixture<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GELS, batched__float)
{
    test_fixture<true, true, float>();
}

TEST_P(GELS, batched__double)
{
    test_fixture<true, true, double>();
}

TEST_P(GELS, batched__float_complex)
{
    test_fixture<true, true, rocblas_float_complex>();
}

TEST_P(GELS, batched__double_complex)
{
    test_fixture<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GELS, strided_batched__float)
{
    test_fixture<false, true, float>();
}

TEST_P(GELS, strided_batched__double)
{
    test_fixture<false, true, double>();
}

TEST_P(GELS, strided_batched__float_complex)
{
    test_fixture<false, true, rocblas_float_complex>();
}

TEST_P(GELS, strided_batched__double_complex)
{
    test_fixture<false, true, rocblas_double_complex>();
}

// daily_lapack tests normal execution with medium to large sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GELS,
                         Combine(ValuesIn(large_matrix_sizeA_range),
                                 ValuesIn(large_matrix_sizeB_range)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GELS,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));
