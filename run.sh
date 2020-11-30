#SBATCH -p long
#SBATCH -t 5-00:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH -n 5
#SBATCH -g 4
#SBACTH -N 2
#SBATCH --mail-type=END,FAIL

python --version

export PYTHONPATH="$PYTHONPATH:.:foodseg"
python -m pip install -r requirements.txt \
    -f https://download.pytorch.org/whl/cu101/torch_stable.html \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
python foodseg/main.py
