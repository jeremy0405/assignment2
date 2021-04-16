# 코드 설명

    import torch
    import torchvision
    import torch.utils.data as data
    import torchvision.transforms as transforms

    if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         normalize,
         ])
         
    test_set = torchvision.datasets.ImageFolder(root="/content/drive/My Drive/ILSVRC2012_img_val/", transform=transform)
    test_loader = data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=2)        
    
   
    model = torchvision.models.[model_name](pretrained=True).to(device)
    model.eval()
    
    correct_top1 = 0
    total = 0

    correct_top1 = 0
    total = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):

            images = images.to(device)      # [100, 3, 224, 224]
            labels = labels.to(device)      # [100]
            outputs = model(images)

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct_top1 += (pred == labels).sum().item()

            print("step : {} / {}".format(idx + 1, len(test_set)/int(labels.size(0))))
            print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
            
    print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
