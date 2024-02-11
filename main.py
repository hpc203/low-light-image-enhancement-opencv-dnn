import argparse
import cv2
import numpy as np

class PairLIE:
    def __init__(self, modelpath, exposure=0.5):
        # Initialize model
        self.net = cv2.dnn.readNet(modelpath)

        hxw = modelpath.replace('.onnx', '').split('_')[1].split('x')  ###pairlie_HxW.onnx除外
        self.input_height = int(hxw[0])
        self.input_width = int(hxw[1])
        self.exposure = np.array([exposure], dtype=np.float32)
        
    def detect(self, image):   ###你也可以把exposure作为输入参数
        input_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dsize=(
            self.input_width, self.input_height))
        input_image = input_image.astype(np.float32) / 255.0
        blob = cv2.dnn.blobFromImage(input_image)
        self.net.setInput(blob, "input")
        self.net.setInput(cv2.dnn.blobFromImage(self.exposure), "exposure")
        result = self.net.forward(self.net.getUnconnectedOutLayersNames())

        # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
        output_image = np.squeeze(result[0])
        output_image = output_image.transpose(1, 2, 0)
        output_image = output_image * 255
        output_image = np.clip(output_image, 0, 255)
        output_image = output_image.astype(np.uint8)
        output_image = output_image[..., ::-1]
        return output_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str,
                        default='testimgs/1_1.JPG', help="image path")
    parser.add_argument('--modelpath', type=str,
                        default='weights/pairlie_512x512.onnx', help="image path")
    parser.add_argument('--exposure', type=float, default=0.5)
    args = parser.parse_args()

    mynet = PairLIE(args.modelpath, exposure=args.exposure)
    srcimg = cv2.imread(args.imgpath)

    dstimg = mynet.detect(srcimg)
    dstimg = cv2.resize(dstimg, (srcimg.shape[1], srcimg.shape[0]))

    if srcimg.shape[0] > srcimg.shape[1]:
        boundimg = np.zeros((10, srcimg.shape[1], 3), dtype=srcimg.dtype)+255
        combined_img = np.vstack([srcimg, boundimg, dstimg])
    else:
        boundimg = np.zeros((srcimg.shape[0], 10, 3), dtype=srcimg.dtype)+255
        combined_img = np.hstack([srcimg, boundimg, dstimg])
    winName = 'Deep Learning in OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, combined_img)  ###原图和结果图也可以分开窗口显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()
