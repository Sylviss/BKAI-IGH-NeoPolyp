# BKAI-IGH-NeoPolyp

Uncommon dependencies: torchgeometry, segmentation-models-pytorch

run this if you don't sure:

pip install torchgeometry segmentation-models-pytorch

If there's any problem with the code, please create an issue. If there is an issue, it should be about Python or package version. I'm using Python 3.11 with PyTorch 2.2

weight files: https://drive.google.com/file/d/1jmcJWidEAUEHdl-doNnXuAccblYv2Q11/view

the weight should be at the top directory together with the infer.py

commands:

git clone https://github.com/Sylviss/BKAI-IGH-NeoPolyp

cd BKAI-IGH-NeoPolyp

python3 infer.py --image_path image.jpeg
