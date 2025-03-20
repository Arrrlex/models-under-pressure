# As soon as you've logged into your lambda instance, copy this script,
# paste it into the terminal, and follow the instructions to set up your instance.

cat << 'EOF' > setup.sh && chmod +x setup.sh && ./setup.sh && rm setup.sh
#!/bin/bash
set -e

cat << 'KEYS' >> ~/.ssh/authorized_keys
# Alex's public key
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMPl9sU3yu7y0/TOwBR7W9UCL5Os/CM7wgLWRwrnit3L
# Will's public key
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC3QMwhm6skPSTaARnEGnDIpgQXqBukndv95vkpKTXuCz4wXFgyL4885cCZqT8mq28CL5SxEBeDl2zRryFbl9MKviWfMEcG3VA8dbzL8P/C+fqkMdAYk2OOXcINfyBaR83LR5HZ//6fOKx1g0TR4rFL+oF3SZbk9T2Ds6OHfeGnoKRq9BWzOECZF3WCTgEaAVr4PQrxC8OABy+bJEzpLvzSG/0JhMeqdMin17deLwO62QXAA3tN9qvyrLmM0Buu1jmu2NbqXuKCRGgYzipc5JyNKikmntwVJ1v2rjknN0lLaqbd3ZecPbwDsp7kHUjwDqpfKsh33vKFCyI83gquRffv Will Lambda Labs
# Urja's public key
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILkWlemnTwJ7MtdZ3j7pYsOKKHBNSjEtgOaOvzGWpRew
# Urja's other public key
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC6q9vhHXxdRyA1YBkufgRy1cH47/UP3qWEhLC2hg2//a6iMfgxzP2bbZLH1bCllIAE9DGbZ65EdPuaoWLckywpUTBjY/kLCHnd1VdEDVuosV6WPhCLDzTlN9f5833xYrdhDvDIbDH+gBYz6cniSGqlurw8HLSR+UIlsQW0OwtAPZen5g8/fy/WZezn7oSXeK8aSbi1xixy/FzVQuxv5Hcoege0jYFVk/rPKGcIE9erApqB66tUWHy1n0210oHk6jm/SjkFsybye71FTfnNVYKF2hYHtyOsE4ol48JekTavPx4sk82pmjT2p8otweKFeylLwzGukVlP49Inbl4dZ3mF Urja Lambda Labs
# Phil's public key
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILhZTNPVEMnfG5K5mMzMs3POZBe530J3oqygeWPUOoWg john@Philipps-MacBook-Air.local
KEYS

# Prompt for SSH key
echo "Please paste your private SSH key (including BEGIN and END lines):"
mkdir -p ~/.ssh
while IFS= read -r line; do
    [[ "$line" == "-----END"* ]] && { echo "$line"; break; }
    echo "$line"
done > ~/.ssh/github_urja

# Set correct permissions
chmod 400 ~/.ssh/github_urja

# Prompt for Huggingface token
echo "Please enter your Huggingface API token:"
read -r hf_token

# Rest of setup
mkdir -p urja && \
cd urja && \
# Set git config for this specific directory before cloning
git config --global --add includeIf."gitdir:$(pwd)/".path "$(pwd)/.gitconfig" && \
cat << 'GITCONFIG' > .gitconfig
[core]
    sshCommand = "ssh -i ~/.ssh/github_urja"
[user]
    email = urjapawar@gmail.com
    name = Urja Pawar
GITCONFIG

git clone git@github.com:Arrrlex/models-under-pressure.git && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
cd models-under-pressure && \
echo "HF_TOKEN=$hf_token" > .env && \
uv sync && \
uv run pre-commit install
EOF
