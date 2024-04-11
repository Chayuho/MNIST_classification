# MNIST Dataset Classification

MNIST 손글씨 숫자를 분류하는 신경망 모델을 구현
  1. LeNet5
  2. Custom MLP

## 파일 설명

- `main.py`: 모델을 학습 및 테스트
- `dataset.py`: MNIST 데이터셋 로드 및 전처리
- `model.py`: LeNet-5 및 Custom MLP 모델을 구현

## main.py 사용자 입력 변수
- train(model_name, train_loader, test_loader, device, criterion, optimizer, epochs)
  - model_name : LeNet5, CustomMLP_model
  - train_loader : LeNet5_train_loader, CST_MLP_train_loader
  - test_loader : LeNet5_test_loader, CST_MLPLeNet5_test_loader
  - device : "cpu", "cuda"
  - epochs : 10, 20, 30, ...

## 실험결과
### LeNet5
![LeNet5_train_loss](https://github.com/Chayuho/MNIST_classification/assets/94342487/c4fd892b-0331-49ca-bb35-2bc29aac7e3b)
![LeNet5_train_acc](https://github.com/Chayuho/MNIST_classification/assets/94342487/7cb3e2d6-2ef5-4c2c-a81a-59d264e08201)
![LeNet5_test_loss](https://github.com/Chayuho/MNIST_classification/assets/94342487/40e9b20c-8ce5-413c-b037-107f5da20b77)
![LeNet5_test_acc](https://github.com/Chayuho/MNIST_classification/assets/94342487/903d67c5-fb94-4094-b4f9-d0b41c3a273d)

### Custom MLP
![CustomMLP_train_loss](https://github.com/Chayuho/MNIST_classification/assets/94342487/0faf4433-53fd-4977-9994-71298bc85105)
![CustomMLP_train_acc](https://github.com/Chayuho/MNIST_classification/assets/94342487/bb8fd77e-608e-47f6-9eaa-bbfa707eb9e6)
![CustomMLP_test_loss](https://github.com/Chayuho/MNIST_classification/assets/94342487/5535032e-82aa-4de5-a670-953928801dda)
![CustomMLP_test_acc](https://github.com/Chayuho/MNIST_classification/assets/94342487/0e776def-3fc3-413a-a361-575a1f1de70a)

### Test Loss 및 Accuracy
| MNIST     | Loss      | Accuracy  |
|-----------|-----------|-----------|
| LeNet5    | 0.096     | 0.973     |
| Custom MLP| 0.046     | 0.986     |

## 요구사항

- requirements.txt

## 참고

- 이 프로젝트는 MNIST 데이터셋을 사용하여 기본적인 딥러닝 모델을 구현하는 방법을 익히기 위한 목적임
- 실제 환경에서는 더욱 복잡한 모델과 데이터셋을 사용해야할 수 있음
