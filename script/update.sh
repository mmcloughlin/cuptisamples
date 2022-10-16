#!/bin/bash -ex

set -exuo pipefail

CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64"

# Options.
function usage() {
    echo "Usage: ${0} [-i <install_dir>]"
    exit 2
}

install_dir="."
while getopts "i:" opt; do
    case "${opt}" in
        p) install_dir="${OPTARG}" ;;
        *) usage ;;
    esac
done

# Temporary working location.
work_dir=$(mktemp -d)

# Setup custom apt configuration.
apt_root="${work_dir}/apt"

mkdir -p "${apt_root}/etc/apt/apt.conf.d"
mkdir -p "${apt_root}/etc/apt/sources.list.d"
mkdir -p "${apt_root}/etc/apt/preferences.d"
mkdir -p "${apt_root}/var/lib/apt/lists/partial"
mkdir -p "${apt_root}/var/lib/dpkg"
touch "${apt_root}/var/lib/dpkg/status"

export APT_CONFIG="${apt_root}/etc/apt/apt.conf"
echo "Dir \"${apt_root}\";" > "${APT_CONFIG}"

# Configure CUDA repository.
apt-key adv --fetch-keys "${CUDA_REPO}/3bf863cc.pub"
echo "deb ${CUDA_REPO} /" > "${apt_root}/etc/apt/sources.list.d/cuda.list"

# Update packages.
apt-get update

# Select the latest CUPTI packages.
cuda_deps="${work_dir}/cuda_deps.txt"
apt-cache depends \
    --recurse \
    --no-recommends \
    --no-suggests \
    --no-conflicts \
    --no-breaks \
    --no-replaces \
    --no-enhances \
    cuda | awk '$1=="Depends:" && $2~/^cuda-/ {print $2}' > "${cuda_deps}"

# Download it.
pkg_dir="${work_dir}/pkg"
mkdir -p "${pkg_dir}"

for pkg in cuda-cupti-dev cuda-documentation; do
    pkg_name=$(grep "${pkg}" "${cuda_deps}")
    (cd "${pkg_dir}" && apt-get download "${pkg_name}")
done

for deb in "${pkg_dir}"/*.deb; do
    dpkg --extract "${deb}" "${pkg_dir}"
done

# Install samples.
rm -rf "${install_dir}/samples"
cp -r "${pkg_dir}"/usr/local/cuda-*/{extras/CUPTI/samples,EULA.txt,README} "${install_dir}"
mv "${install_dir}"/{README,LICENSE}
