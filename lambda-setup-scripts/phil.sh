cat << 'EOF' > setup.sh && chmod +x setup.sh && ./setup.sh && rm setup.sh
#!/bin/bash
set -e

# Prompt for SSH key
echo "Please paste your private SSH key (including BEGIN and END lines):"
mkdir -p ~/.ssh
while IFS= read -r line; do
    [[ "$line" == "-----END"* ]] && { echo "$line"; break; }
    echo "$line"
done > ~/.ssh/github_phil

# Determine key type and move to appropriate filename
if grep -q "BEGIN OPENSSH PRIVATE KEY" ~/.ssh/temp_key; then
    mv ~/.ssh/temp_key ~/.ssh/github_phil
elif grep -q "BEGIN RSA PRIVATE KEY" ~/.ssh/temp_key; then
    mv ~/.ssh/temp_key ~/.ssh/github_phil
else
    echo "Unrecognized key format"
    rm ~/.ssh/temp_key
    exit 1
fi

# Set correct permissions
chmod 400 ~/.ssh/github_phil

# Create SSH config for the specific key
cat << 'SSHCONFIG' >> ~/.ssh/config
Host github.com-phil
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_phil
SSHCONFIG

# Prompt for Huggingface token
echo "Please enter your Huggingface API token:"
read -r hf_token

# Rest of setup
mkdir -p phil && \
cd phil && \
git clone git@github.com-phil:Arrrlex/models-under-pressure.git && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
cd models-under-pressure && \
echo "HF_TOKEN=$hf_token" > .env && \
uv sync && \
uv run pre-commit install && \
git config --global user.email "philipp.blandfort@rtl-extern.de" && \
git config --global user.name "Philipp Blandfort" && \
# Set git config for this specific directory
cd .. && \
git config --global --add includeIf."gitdir:$(pwd)/".path "$(pwd)/.gitconfig" && \
cat << 'GITCONFIG' > .gitconfig
[core]
    sshCommand = "ssh -i ~/.ssh/github_phil -F /dev/null"
[user]
    email = philipp.blandfort@rtl-extern.de
    name = Philipp Blandfort
GITCONFIG
EOF
