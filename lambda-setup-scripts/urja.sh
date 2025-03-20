cat << 'EOF' > setup.sh && chmod +x setup.sh && ./setup.sh && rm setup.sh
#!/bin/bash
set -e

# Prompt for SSH key
echo "Please paste your private SSH key (including BEGIN and END lines):"
mkdir -p ~/.ssh
while IFS= read -r line; do
    [[ "$line" == "-----END"* ]] && { echo "$line"; break; }
    echo "$line"
done > ~/.ssh/github_urja

# Set correct permissions
chmod 400 ~/.ssh/github_urja

# Create SSH config for the specific key
cat << 'SSHCONFIG' >> ~/.ssh/config
Host github.com-urja
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_urja
SSHCONFIG

# Prompt for Huggingface token
echo "Please enter your Huggingface API token:"
read -r hf_token

# Rest of setup
mkdir -p urja && \
cd urja && \
git clone git@github.com-urja:Arrrlex/models-under-pressure.git && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
cd models-under-pressure && \
echo "HF_TOKEN=$hf_token" > .env && \
uv sync && \
uv run pre-commit install && \
git config --global user.email "urjapawar@gmail.com" && \
git config --global user.name "Urja Pawar" && \
# Set git config for this specific directory
cd .. && \
git config --global --add includeIf."gitdir:$(pwd)/".path "$(pwd)/.gitconfig" && \
cat << 'GITCONFIG' > .gitconfig
[core]
    sshCommand = "ssh -i ~/.ssh/github_urja -F /dev/null"
[user]
    email = urjapawar@gmail.com
    name = Urja Pawar
GITCONFIG
EOF
