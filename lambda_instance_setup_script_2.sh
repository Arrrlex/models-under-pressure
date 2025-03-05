chmod 400 ~/.ssh/id_ed25519
git clone git@github.com:Arrrlex/models-under-pressure.git
curl -LsSf https://astral.sh/uv/install.sh | sh
cd models-under-pressure
uv sync
uv run pre-commit install
uv run huggingface-cli login
