cat << 'EOF' > setup.sh && chmod +x setup.sh && ./setup.sh && rm setup.sh
#!/bin/bash
set -e

# Prompt for SSH key
echo "Please paste your private SSH key (including BEGIN and END lines):"
mkdir -p ~/.ssh
while IFS= read -r line; do
    [[ "$line" == "-----END"* ]] && { echo "$line"; break; }
    echo "$line"
done > ~/.ssh/github_will

# Set correct permissions
chmod 400 ~/.ssh/github_will

# Prompt for Huggingface token
echo "Please enter your Huggingface API token:"
read -r hf_token

# Rest of setup
mkdir -p will && \
cd will && \
# Set git config for this specific directory before cloning
git config --global --add includeIf."gitdir:$(pwd)/".path "$(pwd)/.gitconfig" && \
cat << 'GITCONFIG' > .gitconfig
[core]
    sshCommand = "ssh -i ~/.ssh/github_will"
[user]
    email = williamjamesbankes@gmail.com
    name = William Bankes
GITCONFIG

git clone git@github.com:Arrrlex/models-under-pressure.git && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
cd models-under-pressure && \
echo "HF_TOKEN=$hf_token" > .env && \
uv sync && \
uv run pre-commit install
EOF
