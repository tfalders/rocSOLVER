#!/bin/sh

# run this script in develop after creating release-staging branch for feature-complete date
# Edit script to bump versions for new development cycle/release.

# for rocSOLVER version string
OLD_ROCSOLVER_VERSION="3.25.0"
NEW_ROCSOLVER_VERSION="3.26.0"
sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt

# for rocBLAS package requirements
OLD_ROCBLAS_VERSION_DOWN="4.1"
NEW_ROCBLAS_VERSION_DOWN="4.2"
OLD_ROCBLAS_VERSION_UP="4.2"
NEW_ROCBLAS_VERSION_UP="4.3"
sed -i "s/${OLD_ROCBLAS_VERSION_UP}/${NEW_ROCBLAS_VERSION_UP}/g" CMakeLists.txt
sed -i "s/${OLD_ROCBLAS_VERSION_DOWN}/${NEW_ROCBLAS_VERSION_DOWN}/g" CMakeLists.txt

