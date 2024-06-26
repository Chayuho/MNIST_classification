# MNIST Dataset Classification
MNIST 손글씨 숫자를 분류하는 신경망 모델 구현
  1. LeNet5
  2. Custom MLP
## 파일 설명
- `main.py`: 모델을 학습 및 테스트
- `dataset.py`: MNIST 데이터셋 로드 및 전처리
- `model.py`: LeNet-5 및 Custom MLP 모델을 구현

## 모델 설명
### LeNet5 (Num of parameters : 61,706)
![Lenet5](https://github.com/Chayuho/MNIST_classification/assets/94342487/f8d850d2-0329-47b8-b528-cce4a6177682)
- dropout : 0.3
- Tanh

### Custom MLP (Num of parameters : 61,745)
- Number of Layers : 3
  - Layer 1 : img_size (1, 32, 32) -> (1, 32*32) * 59
  - Layer 2 : 59 * 18
  - Layer 3 : 18 * num_class(10)
- dropout : 0.3
- ReLU
  
## main.py 사용자 입력 변수
- train(model_name, train_loader, test_loader, device, criterion, optimizer, epochs)
  - model_name : LeNet5, CustomMLP_model
  - train_loader : LeNet5_train_loader, CST_MLP_train_loader
  - test_loader : LeNet5_test_loader, CST_MLPLeNet5_test_loader
  - device : "cpu", "cuda"
  - epochs : 10, 20, 30, ...
  - Learning rate : 0.01
  - Optimizer : SGD (momentum = 0.9)

## 실험결과
### LeNet5
![NeNet5_train_loss](https://github.com/Chayuho/MNIST_classification/assets/94342487/f30ecf4d-bf72-4bee-9ad8-3059bdf12467s=0.3)
![NeNet5_train_acc](https://github.com/Chayuho/MNIST_classification/assets/94342487/ac3438e4-8830-4db9-824c-af833821c059)
![NeNet5_test_loss](https://github.com/Chayuho/MNIST_classification/assets/94342487/06e16218-7f63-40a8-a0ba-58feecee890e)
![NeNet5_test_acc](https://github.com/Chayuho/MNIST_classification/assets/94342487/564789d0-2d18-4b53-9c40-c65479973cbc)

### Custom MLP
![CustomMLP_train_loss](https://github.com/Chayuho/MNIST_classification/assets/94342487/0faf4433-53fd-4977-9994-71298bc85105)
![CustomMLP_train_acc](https://github.com/Chayuho/MNIST_classification/assets/94342487/bb8fd77e-608e-47f6-9eaa-bbfa707eb9e6)
![CustomMLP_test_loss](https://github.com/Chayuho/MNIST_classification/assets/94342487/5535032e-82aa-4de5-a670-953928801dda)
![CustomMLP_test_acc](https://github.com/Chayuho/MNIST_classification/assets/94342487/0e776def-3fc3-413a-a361-575a1f1de70a)

### Test Loss 및 Accuracy
| MNIST     | Loss      | Accuracy  |
|-----------|-----------|-----------|
| LeNet5    | 0.046     | 0.986     |
| Custom MLP| 0.096     | 0.973     |

## 요구사항
- requirements.txt
## 참고
- 이 프로젝트는 MNIST 데이터셋을 사용하여 기본적인 딥러닝 모델을 구현하는 방법을 익히기 위한 목적임
