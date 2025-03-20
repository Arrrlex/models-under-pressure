cat << 'EOF' > setup.sh && chmod +x setup.sh && ./setup.sh && rm setup.sh
#!/bin/bash
set -e

# Prompt for SSH key
echo "Please paste your private SSH key (including BEGIN and END lines):"
mkdir -p ~/.ssh
while IFS= read -r line; do
    [[ "$line" == "-----END"* ]] && { echo "$line"; break; }
    echo "$line"
done > ~/.ssh/temp_key

# Determine key type and move to appropriate filename
if grep -q "BEGIN OPENSSH PRIVATE KEY" ~/.ssh/temp_key; then
    mv ~/.ssh/temp_key ~/.ssh/id_ed25519
elif grep -q "BEGIN RSA PRIVATE KEY" ~/.ssh/temp_key; then
    mv ~/.ssh/temp_key ~/.ssh/id_rsa
else
    echo "Unrecognized key format"
    rm ~/.ssh/temp_key
    exit 1
fi

# Set correct permissions
chmod 400 ~/.ssh/id_*

# Prompt for Huggingface token
echo "Please enter your Huggingface API token:"
read -r hf_token

# Rest of setup
mkdir -p urja && \
cd urja && \
git clone git@github.com:Arrrlex/models-under-pressure.git && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
cd models-under-pressure && \
echo "HF_TOKEN=$hf_token" > .env && \
uv sync && \
uv run pre-commit install && \
git config --global user.email "urjapawar@gmail.com" && \
git config --global user.name "Urja Pawar"
EOF
