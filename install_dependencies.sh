#!/bin/bash

# Clone AMR repository (not a package, so we clone it directly)
echo "Cloning AMR repository..."
git clone https://github.com/RikVN/AMR.git

# Get the ucca module code (there are issues with compatibility with other modules, so we download it separately)
# https://pypi.org/project/UCCA/#description
ucca_dir="./scripts/ucca"
mkdir -p "${ucca_dir}"
wget https://files.pythonhosted.org/packages/4c/a5/a42028dcc21c3d1ddaf609daa57863362bc88bff7072b5c96550aa06ca5f/UCCA-1.3.11.tar.gz -O ucca.tar.gz
# extract just the contents of UCCA-1.3.11/ucca
tar --strip-components=2 -xzf ucca.tar.gz -C "${ucca_dir}" UCCA-1.3.11/ucca
rm ucca.tar.gz

# Create data directory (if it doesn't exist)
mkdir -p data
