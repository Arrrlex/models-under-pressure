#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Ensure the script is being run from the user's home directory
cd ~

# 1. Update .gitconfig with include paths
echo "Updating ~/.gitconfig with include paths..."
cat << EOF >> ~/.gitconfig
[includeIf "gitdir:lasr-fs-utah-1/alex/"]
    path = lasr-fs-utah-1/alex/.gitconfig
[includeIf "gitdir:lasr-fs-utah-1/urja/"]
    path = lasr-fs-utah-1/urja/.gitconfig
[includeIf "gitdir:lasr-fs-utah-1/will/"]
    path = lasr-fs-utah-1/will/.gitconfig
[includeIf "gitdir:lasr-fs-utah-1/phil/"]
    path = lasr-fs-utah-1/phil/.gitconfig
EOF

# 2. Add HF_HOME export to .bashrc
echo "Adding HF_HOME export to .bashrc..."
if ! grep -q "export HF_HOME=~/lasr-fs-utah-1/hf/" ~/.bashrc; then
    echo "export HF_HOME=~/lasr-fs-utah-1/hf/" | cat - ~/.bashrc > temp && mv temp ~/.bashrc
fi

# 3. Install uv (micromamba replacement)
echo "Installing uv..."
if ! command_exists uv; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "Setup complete!"
