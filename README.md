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
![NeNet5_train_loss](https://github.com/uuuuhoo/LeNet5-CustomMLP/assets/166509489/e70456ac-34d0-4ede-83af-43509c87f77b){:width="50%" height="50%"}
![NeNet5_train_acc](https://github.com/uuuuhoo/LeNet5-CustomMLP/assets/166509489/287c05a8-edb9-4370-90b9-15e657b9dfc3){:width="50%" height="50%"}
![NeNet5_test_loss](https://github.com/uuuuhoo/LeNet5-CustomMLP/assets/166509489/a3c8bb4e-cea8-47a8-b587-fae4baeb6276){:width="50%" height="50%"}
![NeNet5_test_acc](https://github.com/uuuuhoo/LeNet5-CustomMLP/assets/166509489/ecd07bd7-232c-4032-bccc-a595a7952890){:width="50%" height="50%"}
![CustomMLP_train_loss](https://github.com/uuuuhoo/LeNet5-CustomMLP/assets/166509489/89bc8712-f222-473e-ab8a-270dab3bcc2d){:width="50%" height="50%"}
![CustomMLP_train_acc](https://github.com/uuuuhoo/LeNet5-CustomMLP/assets/166509489/ea8a8d76-2ea8-4186-be2d-13142a5e4bb8){:width="50%" height="50%"}
![CustomMLP_test_loss](https://github.com/uuuuhoo/LeNet5-CustomMLP/assets/166509489/c539e6ff-d9b0-4b3a-8ae7-598beaa78ed5){:width="50%" height="50%"}
![CustomMLP_test_acc](https://github.com/uuuuhoo/LeNet5-CustomMLP/assets/166509489/7ae0f323-f296-445e-8efe-1792f1598231){:width="50%" height="50%"}

## 요구사항

- requirements.txt

## 참고

- 이 프로젝트는 MNIST 데이터셋을 사용하여 기본적인 딥러닝 모델을 구현하는 방법을 익히기 위한 목적임
- 실제 환경에서는 더욱 복잡한 모델과 데이터셋을 사용해야할 수 있음
