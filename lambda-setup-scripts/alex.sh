cat << 'EOF' > setup.sh && chmod +x setup.sh && ./setup.sh && rm setup.sh
#!/bin/bash
set -e

# Prompt for SSH key
echo "Please paste your private SSH key (including BEGIN and END lines):"
mkdir -p ~/.ssh
while IFS= read -r line; do
    [[ "$line" == "-----END"* ]] && { echo "$line"; break; }
    echo "$line"
done > ~/.ssh/github_alex


# Set correct permissions
chmod 400 ~/.ssh/github_alex

# Create SSH config for the specific key
cat << 'SSHCONFIG' >> ~/.ssh/config
Host github.com-alex
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_alex
SSHCONFIG

# Prompt for Huggingface token
echo "Please enter your Huggingface API token:"
read -r hf_token

# Rest of setup
mkdir -p alex && \
cd alex && \
git clone git@github.com-alex:Arrrlex/models-under-pressure.git && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
cd models-under-pressure && \
echo "HF_TOKEN=$hf_token" > .env && \
uv sync && \
uv run pre-commit install && \
git config --global user.email "me+github@alexmck.com" && \
git config --global user.name "Alex McKenzie" && \
# Set git config for this specific directory
cd .. && \
git config --global --add includeIf."gitdir:$(pwd)/".path "$(pwd)/.gitconfig" && \
cat << 'GITCONFIG' > .gitconfig
[core]
    sshCommand = "ssh -i ~/.ssh/github_alex -F /dev/null"
[user]
    email = me+github@alexmck.com
    name = Alex McKenzie
GITCONFIG
EOF
