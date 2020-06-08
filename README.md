Download weights here and place them in model_data/
- [yolov2.weights](https://www.dropbox.com/s/57zhd75mmmc5olf/yolov2.weights?dl=0)
- [MobileNetSSD_deploy.caffemodel](https://www.dropbox.com/s/d7pxo7kw67zb0e1/MobileNetSSD_deploy.caffemodel?dl=0)

To execute code, open CMD and move to the root directory of the project. Execute this command
python src/main.py --model model_data/MobileNetSSD_deploy.caffemodel --config model_data/MobileNetSSD_deploy.prototxt --output out/sample_output.avi --classes model_data/MobileNet_classes.txt
