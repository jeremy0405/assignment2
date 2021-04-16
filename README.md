# 코드 실행 결과
Alexnet   Top-1 Accuracy : 56.52%
![alexnet](https://user-images.githubusercontent.com/81368630/115024741-76150100-9efb-11eb-99e7-96e012f9dafa.jpg)

VGG16     Top-1 Accuracy : 71.59%
![vgg](https://user-images.githubusercontent.com/81368630/115024753-7ad9b500-9efb-11eb-93de-eefcc6b7e324.jpg)

ResNet18  Top-1 Accuracy : 69.76%
![resnet](https://user-images.githubusercontent.com/81368630/115024766-7f05d280-9efb-11eb-81f6-eb8c80af765c.jpg)

GoogLeNet Top-1 Accuracy : 69.78%
![googlenet](https://user-images.githubusercontent.com/81368630/115024791-8af19480-9efb-11eb-91b6-d2d9fe1b6d52.jpg)


# 코드 설명

    import torch
    import torchvision
    import torch.utils.data as data
    import torchvision.transforms as transforms

    if __name__ == "__main__":
    #gpu 또는 cpu를 사용할 것인지 정한다.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #이미지를 정규화하기 위해서 mean함수와 std함수를 이용한다.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    #이미지를 256*256으로 resize후 224*224로 centercrop 후 정규화한다.
    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         normalize,
         ])
         
    #구글드라이브에 분류하여 올린 2012 imagenet validation set을 dataset으로서 test_set으로 지정 후 test_loader에 test_set을 배치사이즈 100, 랜덤으로 셔플 후 넣어준다.    
    test_set = torchvision.datasets.ImageFolder(root="/content/drive/My Drive/ILSVRC2012_img_val/", transform=transform)    
    test_loader = data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=2)        
    
    #model_name 에 테스트할 pretrained model(alexnet, vgg16, resnet18, googlenet)을 model에 넣어준 후 model의 구조를 확인한다.
    model = torchvision.models.[model_name](pretrained=True).to(device)
    model.eval()
    
    #Top-1 Accuracy를 확인하기 위해 변수 correct_top1 및 total을 지정한다.
    correct_top1 = 0
    total = 0

    #   
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):

            #100장씩 batch_size가 되어 구별된 images와 labels을 변수에 넣어준 후 outputs을 통해 모델에 넣어준다.
            images = images.to(device)      # [100, 3, 224, 224]
            labels = labels.to(device)      # [100]
            outputs = model(images)
            
            #예측값과 label이 동일하면 맞춘것이므로 correct_top1을 증가하여 맞춘 개수를 세어준다.
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct_top1 += (pred == labels).sum().item()

            #correct_top1 변수와 total을 이용하여 top-1 percentage (= top-1 accuracy)를 step마다 출력한다.
            print("step : {} / {}".format(idx + 1, len(test_set)/int(labels.size(0))))
            print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
            
    #모든 validation set을 model을 통해 검증한 후 top-1 percentage (= top-1 accuracy)를 출력한다.         
    print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
