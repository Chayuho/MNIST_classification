# MNIST Dataset Classification

MNIST 손글씨 숫자를 분류하는 신경망 모델을 구현하는 것을 목표로 합니다.
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


## 요구사항

- requirements.txt

## 참고

- 이 프로젝트는 MNIST 데이터셋을 사용하여 기본적인 딥러닝 모델을 구현하는 방법을 익히기 위한 목적임
- 실제 환경에서는 더욱 복잡한 모델과 데이터셋을 사용해야할 수 있음
