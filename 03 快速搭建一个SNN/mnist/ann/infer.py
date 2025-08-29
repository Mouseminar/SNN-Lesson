import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import MLP


class MNISTInference:
    def __init__(self, model_path='checkpoints/mnist_mlp.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MLP().to(self.device)
        
        # 加载训练好的模型
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 数据预处理变换（与训练时保持一致）
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        print(f"模型已加载到 {self.device}")
    
    def preprocess_image(self, image_path):
        """预处理单张图像"""
        try:
            # 加载图像
            image = Image.open(image_path)
            
            # 如果是RGBA，转换为RGB
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # 应用变换
            image_tensor = self.transform(image).unsqueeze(0)  # 添加batch维度
            return image_tensor.to(self.device)
        
        except Exception as e:
            print(f"图像预处理失败: {e}")
            return None
    
    def predict(self, image_path, show_confidence=True):
        """对单张图像进行预测"""
        # 预处理图像
        image_tensor = self.preprocess_image(image_path)
        if image_tensor is None:
            return None
        
        # 推理
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
        
        if show_confidence:
            print(f"预测结果: {predicted_class}")
            print(f"置信度: {confidence_score:.4f}")
            
            # 显示所有类别的概率
            probs = probabilities.cpu().numpy()[0]
            print("\n各数字的概率:")
            for i, prob in enumerate(probs):
                print(f"数字 {i}: {prob:.4f}")
        
        return predicted_class, confidence_score
    
    def visualize_prediction(self, image_path):
        """可视化预测结果"""
        try:
            # 加载原始图像
            image = Image.open(image_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # 进行预测
            predicted_class, confidence = self.predict(image_path, show_confidence=False)
            
            # 显示图像和预测结果
            plt.figure(figsize=(8, 4))
            
            # 显示原始图像
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            plt.title(f"输入图像")
            plt.axis('off')
            
            # 显示预处理后的图像
            processed_image = self.preprocess_image(image_path)
            plt.subplot(1, 2, 2)
            plt.imshow(processed_image.cpu().squeeze().numpy(), cmap='gray')
            plt.title(f"预测: {predicted_class}\n置信度: {confidence:.4f}")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"可视化失败: {e}")
            return None, None


def test_with_sample():
    """使用MNIST测试集中的样本进行测试"""
    from torchvision import datasets
    
    # 加载测试数据
    test_dataset = datasets.MNIST('data', train=False, download=True)
    
    # 随机选择一张图像
    import random
    idx = random.randint(0, len(test_dataset)-1)
    image, true_label = test_dataset[idx]
    
    # 保存为临时文件
    image.save('temp_test.png')
    
    # 进行推理
    inference = MNISTInference()
    print(f"真实标签: {true_label}")
    predicted_class, confidence = inference.predict('temp_test.png')
    
    # 可视化
    inference.visualize_prediction('temp_test.png')
    
    # 清理临时文件
    import os
    os.remove('temp_test.png')


if __name__ == '__main__':
    # 示例用法
    print("MNIST 手写数字识别推理")
    print("1. 使用测试集样本进行测试")
    print("2. 使用自定义图像进行预测")
    
    choice = input("请选择 (1/2): ")
    
    if choice == '1':
        test_with_sample()
    elif choice == '2':
        image_path = input("请输入图像路径: ")
        try:
            inference = MNISTInference()
            inference.visualize_prediction(image_path)
        except Exception as e:
            print(f"推理失败: {e}")
    else:
        print("无效选择")