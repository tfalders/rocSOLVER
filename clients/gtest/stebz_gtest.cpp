/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_stebz.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> stebz_tuple;

// each size_range vector is a {n, ord}
// if ord = 1, then order eigenvalues by blocks
// if ord = 0, then order eigenvalues of the entire matrix

// each ops_range vector is a {rng, vl, vu, il, iu}
// if rng = 0, then find all eigenvalues
// if rng = 1, then find eigenavlues in (vl, vu]
// if rng = 2, then find the il-th to the iu-th eigenvalue

// Note: all tests are prepared with diagonally dominant matrices that have random diagonal
// elements in [-20, -11] U [11, 20], and off-diagonal elements in [-4, 5].
// Thus, all eigenvalues should be in [-30, 30]

// case when n == 0, ord == 0, and rng == 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 0},
    // invalid
    {-1, 0},
    // normal (valid) samples
    {15, 0},
    {20, 1}};
const vector<vector<int>> ops_range = {
    // always invalid
    {1, 2, 1, 0, 0},
    {2, 0, 0, 0, -1},
    {2, 0, 0, 1, 25},
    // valid only when n=0
    {2, 0, 0, 1, 0},
    // valid only when n>0
    {2, 0, 0, 1, 5},
    {2, 0, 0, 1, 15},
    {2, 0, 0, 7, 12},
    // always valid samples
    {0, 0, 0, 0, 0},
    {1, -15, -5, 0, 0},
    {1, -5, 5, 0, 0},
    {1, 5, 15, 0, 0}};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {{120, 1}, {256, 0}, {350, 1}, {512, 0}};
const vector<vector<int>> large_ops_range = {{0, 0, 0, 0, 0}, {1, -15, 15, 0, 0}, {2, 0, 0, 50, 75}};

Arguments stebz_setup_arguments(stebz_tuple tup)
{
    Arguments arg;

    vector<int> size = std::get<0>(tup);
    vector<int> op = std::get<1>(tup);

    arg.set<rocblas_int>("n", size[0]);
    arg.set<char>("order", (size[1] == 0 ? 'E' : 'B'));

    arg.set<char>("range", (op[0] == 0 ? 'A' : (op[0] == 1 ? 'V' : 'I')));
    arg.set<double>("vlow", op[1]);
    arg.set<double>("vup", op[2]);
    arg.set<rocblas_int>("ilow", op[3]);
    arg.set<rocblas_int>("iup", op[4]);

    // always use max accuracy for the tests
    arg.set<double>("abstol", 0);

    arg.timing = 0;

    return arg;
}

class STEBZ : public ::TestWithParam<stebz_tuple>
{
protected:
    STEBZ() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = stebz_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("order") == 'E'
           && arg.peek<char>("range") == 'A')
            testing_stebz_bad_arg<T>();

        testing_stebz<T>(arg);
    }
};

// non-batch tests

TEST_P(STEBZ, __float)
{
    run_tests<float>();
}

TEST_P(STEBZ, __double)
{
    run_tests<double>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         STEBZ,
                         Combine(ValuesIn(large_size_range), ValuesIn(large_ops_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, STEBZ, Combine(ValuesIn(size_range), ValuesIn(ops_range)));
