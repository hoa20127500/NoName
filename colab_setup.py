"""Colab install helper. Run this *instead of* pip install torch_scatter.

Do not compile torch_scatter on Colab Python 3.13 — pip falls back to a
source tarball and can sit on "Building wheels" for 30+ minutes.

Also do not `pip install dgl` from PyPI: that resolves to dgl==0.1.3, which is
not the Deep Graph Library this code needs (and breaks on Python 3.13).
Official DGL only publishes wheels for Python 3.8–3.12.

Usage (one Colab cell):

    %run colab_setup.py
"""
import subprocess
import sys


def main():
    print('Python', sys.version.split()[0])
    print('Skipping torch_scatter (now implemented with torch.scatter_reduce).')
    subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'dgl'])
    if sys.version_info >= (3, 13):
        print(
            '\nColab Python 3.13 cannot install official DGL wheels.\n'
            'Use condacolab to get Python 3.11, then install DGL:\n\n'
            '    !pip install -q condacolab\n'
            '    import condacolab\n'
            '    condacolab.install()  # runtime restarts once\n\n'
            'After restart:\n\n'
            '    import torch\n'
            '    print(torch.__version__, torch.version.cuda)\n'
            '    !pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html\n'
        )
        return
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', 'dgl',
        '-f', 'https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html',
    ])


if __name__ == '__main__':
    main()
