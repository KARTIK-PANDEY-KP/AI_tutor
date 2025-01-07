
# Step 1: Install python3-venv if needed
sudo apt install python3.10-venv

# Step 2: Create the virtual environment
python3 -m venv sb

# Step 3: Activate the virtual environment
source sb/bin/activate

pip install --upgrade pip
pip install notebook ipywidgets ipykernel tqdm python-dotenv

python -m ipykernel install --user --name=sb --display-name "Python (sb)"
sudo apt install zip tmux

# For H100 remove the command below
# sudo apt-get update
# sudo apt-get install python3-dev

pip install sycamore-ai[pinecone]
sudo apt-get install poppler-utils
pip install "sycamore-ai[local-inference]"
