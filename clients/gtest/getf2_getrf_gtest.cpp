/* ************************************************************************
 * Copyright (c) 2020-2023 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getf2_getrf.hpp"
#include "testing_getf2_getrf_npvt.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> getrf_tuple;

// each matrix_size_range vector is a {m, lda, singular}
// if singular = 1, then the used matrix for the tests is singular

// case when m = n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {{24000, 24000, 0}};

const vector<int> n_size_range = {24000};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 0},
    {640, 640, 1},
    {1000, 1024, 0},
};

const vector<int> large_n_size_range = {
    45, 64, 520, 1024, 2000,
};

Arguments getrf_setup_arguments(getrf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    int n_size = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m", matrix_size[0]);
    arg.set<rocblas_int>("n", n_size);
    arg.set<rocblas_int>("lda", matrix_size[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;
    arg.singular = matrix_size[2];

    return arg;
}

template <bool BLOCKED>
class GETF2_GETRF : public ::TestWithParam<getrf_tuple>
{
protected:
    GETF2_GETRF() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getrf_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 0)
            testing_getf2_getrf_bad_arg<BATCHED, STRIDED, BLOCKED, T>();

        arg.batch_count = 1;
        if(arg.singular == 1)
            testing_getf2_getrf<BATCHED, STRIDED, BLOCKED, T>(arg);

        arg.singular = 0;
        testing_getf2_getrf<BATCHED, STRIDED, BLOCKED, T>(arg);
    }
};

template <bool BLOCKED>
class GETF2_GETRF_NPVT : public ::TestWithParam<getrf_tuple>
{
protected:
    GETF2_GETRF_NPVT() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getrf_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 0)
            testing_getf2_getrf_npvt_bad_arg<BATCHED, STRIDED, BLOCKED, T>();

        arg.batch_count = 1;
        if(arg.singular == 1)
            testing_getf2_getrf_npvt<BATCHED, STRIDED, BLOCKED, T>(arg);

        arg.singular = 0;
        testing_getf2_getrf_npvt<BATCHED, STRIDED, BLOCKED, T>(arg);
    }
};

class GETF2 : public GETF2_GETRF<false>
{
};

class GETRF : public GETF2_GETRF<true>
{
};

class GETF2_NPVT : public GETF2_GETRF_NPVT<false>
{
};

class GETRF_NPVT : public GETF2_GETRF_NPVT<true>
{
};

// non-batch tests
TEST_P(GETF2_NPVT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETF2_NPVT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETF2_NPVT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETF2_NPVT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETRF_NPVT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRF_NPVT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRF_NPVT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRF_NPVT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETF2, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETF2, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETF2, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETF2, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETRF, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRF, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRF, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRF, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests
TEST_P(GETF2_NPVT, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETF2_NPVT, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETF2_NPVT, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETF2_NPVT, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GETRF_NPVT, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETRF_NPVT, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETRF_NPVT, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETRF_NPVT, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GETF2, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETF2, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETF2, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETF2, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GETRF, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETRF, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETRF, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETRF, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched cases
TEST_P(GETF2_NPVT, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETF2_NPVT, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETF2_NPVT, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETF2_NPVT, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GETRF_NPVT, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETRF_NPVT, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETRF_NPVT, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETRF_NPVT, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GETF2, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETF2, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETF2, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETF2, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GETRF, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETRF, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETRF, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETRF, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GETF2_NPVT,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

// INSTANTIATE_TEST_SUITE_P(checkin_lapack,
//                          GETF2_NPVT,
//                          Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GETRF_NPVT,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF_NPVT,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GETF2,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

// INSTANTIATE_TEST_SUITE_P(checkin_lapack,
//                          GETF2,
//                          Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GETRF,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));
