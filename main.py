"""
reference: https://github.com/mrdbourke/pytorch-apple-silicon/blob/main/01_cifar10_tinyvgg.ipynb
tests the training speed of TinyVGG on cifar10 dataset
"""

import os
import time
import random
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision import transforms

################################################################################

def arg_parser():
    parser = argparse.ArgumentParser(description="PyTorch TinyVGG Training on CIFAR10")
    parser.add_argument("--batch_size", type=int, default=256, help="input batch size for training (default: 256)")
    parser.add_argument("--image_size", type=int, default=32, help="input image size for training (default: 32)")
    parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs to train (default: 5)")
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers to use for data loading (default: 2)")
    # parser.add_argument("--device", type=str, default="cpu", help="device to use for training (default: cpu)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    args = parser.parse_args()
    return args

################################################################################

def main(args):
    # load arguments
    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = (args.image_size, args.image_size)
    NUM_EPOCHS = args.num_epochs
    NUM_WORKERS = args.num_workers
    # DEVICE = "cpu" # args.device
    LR = args.lr
    SEED = args.seed

    # check pytorch version
    print(f"PyTorch version: {torch.__version__}")

    # check MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"is MPS built? {torch.backends.mps.is_built()}")
    print(f"is MPS available? {torch.backends.mps.is_available()}")

    # set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    ############################################################################
    # prepare transforms
    simple_transform = transforms.Compose(
        [
            transforms.Resize(size=IMAGE_SIZE),
            transforms.ToTensor()
        ]
    )

    # load datasets
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        transform=simple_transform,
        download=True
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        transform=simple_transform,
        download=True
    )
    print(f"number of training samples: {len(train_data)})")
    print(f"number of testing samples: {len(test_data)})")

    # create dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"train_dataloader: {train_dataloader}")
    print(f"test_dataloader: {test_dataloader}")

    print(f"1 sample size: {train_data[0][0].shape}")

    # class names
    class_names = train_data.classes
    print(f"class_names: {class_names}")

    # display random images
    def display_random_images(
        dataset: torch.utils.data.dataset.Dataset,
        classes: list = None,
        n: int = 3,
        display_shape: bool = True,
        seed: int = None
    ):
        # set random seed
        if seed:
            random.seed(seed)

        # get random sample indexes
        idx = random.sample(
            range(len(dataset)),
            k=n
        )

        # setup plot
        plt.figure(figsize=(8, 8))

        # loop through samples and display random samples
        for i, sample in enumerate(idx):
            image, label = dataset[sample][0], dataset[sample][1]

            # [C, H, W] -> [H, W, C]
            image = image.permute(1, 2, 0)

            # plot adjusted samples
            plt.subplot(1, n, i+1)
            plt.imshow(image)
            plt.axis(False)
            if classes:
                title = f"class: {classes[label]}"
                if display_shape:
                    title = title + f"\nshape: {image.shape}"
            plt.title(title)
        plt.tight_layout()
        plt.show()

    # display_random_images(
    #     train_data,
    #     n=3,
    #     classes=class_names,
    #     seed=None
    # )

    ############################################################################

    # create model
    class TinyVGG(nn.Module):
        """
        TinyVGG architecture
        """

        def __init__(
            self,
            input_shape: int,
            hidden_units: int,
            output_shape: int
        ) -> None:
            super().__init__()
            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0
                ),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                )
            )
            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    padding=0
                ),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                )
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    in_features=hidden_units*5*5,   # if image size is (32, 32)
                    # in_features=hidden_units*53*53,   # if image size is (224, 224)
                    out_features=output_shape
                )
            )

        def forward(self, x: torch.Tensor):
            x = self.conv_block_1(x)
            x = self.conv_block_2(x)
            x = self.classifier(x)
            return x
        
    # do a dummy forward pass
    model = TinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=3
    )
    print(f"test run: {model(torch.randn(1, 3, 32, 32))}")

    ############################################################################

    # setup training / testing loops
    def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ):
        # train mode
        model.train()

        # loss and accuracy
        loss, acc = 0., 0.

        # loop through data loader data batches
        for batch, (X, y) in enumerate(dataloader):
            # send to device
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # forward pass
            y_pred = model(X)

            # loss
            loss = loss_fn(y_pred, y)
            loss += loss.item()

            # accuracy
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            acc += (y_pred_class == y).sum().item() / len(y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # adjust metrics
        loss = loss / len(dataloader)
        acc = acc / len(dataloader)
        return loss, acc

    def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device
    ):
        # eval mode
        model.eval()

        # loss and accuracy
        loss, acc = 0., 0.

        # inference context manager
        with torch.inference_mode():
            # loop through data loader data batches
            for batch, (X, y) in enumerate(dataloader):
                # send to device
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

                # forward pass
                y_pred_logits = model(X)

                # loss
                loss = loss_fn(y_pred_logits, y)
                loss += loss.item()

                # accuracy
                y_pred_labels = y_pred_logits.argmax(dim=1)
                acc += ((y_pred_labels == y).sum().item() / len(y_pred_labels))

        # adjust metrics
        loss = loss / len(dataloader)
        acc = acc / len(dataloader)
        return loss, acc

    def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        device: torch.device
    ):
        print(f">>> training model '{model.__class__.__name__}' on device '{device}' for {epochs} epochs")

        # results dict
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }

        # loop through epochs
        t0 = time.perf_counter()
        for epoch in range(epochs):
            # train step
            train_loss, train_acc = train_step(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device
            )

            # test step
            test_loss, test_acc = test_step(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device
            )

            # print out
            t1 = time.perf_counter()
            elps = t1 - t0
            print(
                f"epoch: {epoch} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f} | "
                f"time: {elps:.2f} sec"
            )

            # update results
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        return results

    ############################################################################

    # train the model
    torch.manual_seed(42)

    # recreate an instance of TinyVGG
    model = TinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=len(train_data.classes)
    ).to(device)
    print(f"model: {model}")

    # setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=.001
    )

    # train model
    model_results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device
    )

    print(f"model_results: {model_results}")
    print(f"model_results['train_loss']: {model_results['train_loss']}")

    # plot results
    plt.figure(figsize=(7, 5))
    plt.plot(model_results["train_loss"], c="tab:blue", alpha=.7, ls="-", label="tinyvgg_train_loss")
    plt.plot(model_results["test_loss"],  c="tab:blue", alpha=.7, ls=":", label="tinyvgg_test_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss Curves")
    plt.legend()
    plt.show()



    # for comparison, load pre-trained VGG16 and ResNet18
    vgg16 = torchvision.models.vgg16(
        weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1,
        progress=True
    ).to(device)
    res18 = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        progress=True
    ).to(device)
    print(f"vgg16: {vgg16}")
    print(f"res18: {res18}")

    # train vgg16 (pre-trained)
    torch.manual_seed(42)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=vgg16.parameters(),
        lr=.001
    )
    vgg16_results = train(
        model=vgg16,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device
    )

    # train res18 (pre-trained)
    torch.manual_seed(42)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=res18.parameters(),
        lr=.001
    )
    res18_results = train(
        model=res18,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device
    )

    ############################################################################
    # # dict to dataframe
    # model_results = pd.DataFrame(model_results.cpu().numpy())
    # vgg16_results = pd.DataFrame(vgg16_results.cpu().numpy())
    # res18_results = pd.DataFrame(res18_results.cpu().numpy())

    # print dataframe
    print(f"model_results: {model_results}")

    # plot results
    plt.figure(figsize=(7, 5))
    plt.plot(model_results["train_loss"], c="tab:blue", alpha=.7, ls="-", label="tinyvgg_train_loss")
    plt.plot(model_results["test_loss"],  c="tab:blue", alpha=.7, ls=":", label="tinyvgg_test_loss")
    plt.plot(vgg16_results["train_loss"], c="tab:orange", alpha=.7, ls="-", label="vgg16_train_loss")
    plt.plot(vgg16_results["test_loss"],  c="tab:orange", alpha=.7, ls=":", label="vgg16_test_loss")
    plt.plot(res18_results["train_loss"], c="tab:green", alpha=.7, ls="-", label="res18_train_loss")
    plt.plot(res18_results["test_loss"],  c="tab:green", alpha=.7, ls=":", label="res18_test_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss Curves")
    plt.legend()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(model_results["train_acc"], c="tab:blue", alpha=.7, ls="-", label="tinyvgg_train_acc")
    plt.plot(model_results["test_acc"],  c="tab:blue", alpha=.7, ls=":", label="tinyvgg_test_acc")
    plt.plot(vgg16_results["train_acc"], c="tab:orange", alpha=.7, ls="-", label="vgg16_train_acc")
    plt.plot(vgg16_results["test_acc"],  c="tab:orange", alpha=.7, ls=":", label="vgg16_test_acc")
    plt.plot(res18_results["train_acc"], c="tab:green", alpha=.7, ls="-", label="res18_train_acc")
    plt.plot(res18_results["test_acc"],  c="tab:green", alpha=.7, ls=":", label="res18_test_acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracy Curves")
    plt.legend()
    plt.show()

    # print out results
    print(f"\nTinyVGG results;")
    print(f"   train acc: {model_results['train_acc'][-1]:.4f}")
    print(f"   test acc: {model_results['test_acc'][-1]:.4f}")

    print(f"\nVGG16 results;")
    print(f"   train acc: {vgg16_results['train_acc'][-1]:.4f}")
    print(f"   test acc: {vgg16_results['test_acc'][-1]:.4f}")

    print(f"\nResNet18 results;")
    print(f"   train acc: {res18_results['train_acc'][-1]:.4f}")
    print(f"   test acc: {res18_results['test_acc'][-1]:.4f}")

    ############################################################################

    # test with some random images






    exit()

if __name__ == "__main__":
    args = arg_parser()
    main(args)



# if __name__ == "__main__":

    # ################################################################################

    # # Setup hyperparameters
    # BATCH_SIZE = 128 # good for your health: https://twitter.com/ylecun/status/989610208497360896
    # IMAGE_SIZE = (224, 224) # (height, width) smaller images means faster computing 
    # NUM_EPOCHS = 3 # only run for a short period of time... we don't have all day
    # DATASET_NAME = "cifar10" # dataset to use (there are more in torchvision.datasets)
    # MACHINE = "Apple M1 Pro" # change this depending on where you're runing the code
    # NUM_WORKERS = 2 # set number of cores to load data

    # from timeit import default_timer as timer 

    # ################################################################################

    # from torchvision import datasets, transforms
    # from torch.utils.data import DataLoader

    # simple_transform = transforms.Compose([
    #     transforms.Resize(size=IMAGE_SIZE),
    #     transforms.ToTensor()
    # ])

    # # Get Datasets
    # train_data = datasets.CIFAR10(
    #     root="data",
    #     train=True,
    #     transform=simple_transform,
    #     download=True
    # )

    # test_data = datasets.CIFAR10(
    #     root="data",
    #     train=False,
    #     transform=simple_transform,
    #     download=True
    # )

    # print(f"Number of training samples: {len(train_data)}, number of testing samples: {len(test_data)}")

    # # Create DataLoaders
    # train_dataloader = DataLoader(
    #     train_data,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=True
    # )
    # test_dataloader = DataLoader(
    #     test_data,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=True
    # )

    # print(f"train_dataloader: {train_dataloader}")
    # print(f"test_dataloader: {test_dataloader}")

    # print(f"1 sample size: {train_data[0][0].shape}")

    # # class names
    # class_names = train_data.classes
    # print(f"class_names: {class_names}")

    # # Take in a Dataset as well as a list of class names
    # import random
    # from typing import List
    # def display_random_images(
    #         dataset: torch.utils.data.dataset.Dataset,
    #         classes: List[str] = None,
    #         n: int = 10,
    #         display_shape: bool = True,
    #         seed: int = None
    # ):
        
    #     # Adjust display if n too high
    #     if n > 10:
    #         n = 10
    #         display_shape = False
    #         print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
        
    #     # Set random seed
    #     if seed:
    #         random.seed(seed)

    #     # Get random sample indexes
    #     idx = random.sample(range(len(dataset)), k=n)

    #     # Setup plot
    #     plt.figure(figsize=(16, 8))

    #     # Loop through samples and display random samples 
    #     for i, targ_sample in enumerate(idx):
    #         targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

    #         # Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
    #         image = targ_image.permute(1, 2, 0)

    #         # Plot adjusted samples
    #         plt.subplot(1, n, i+1)
    #         plt.imshow(image)
    #         plt.axis("off")
    #         if classes:
    #             title = f"class: {classes[targ_label]}"
    #             if display_shape:
    #                 title = title + f"\nshape: {image.shape}"
    #         plt.title(title)

    # display_random_images(
    #     train_data, 
    #     n=5, 
    #     classes=class_names,
    #     seed=None
    # )

    # ################################################################################

    # # create model (TinyVGG, ref: https://poloclub.github.io/cnn-explainer/)
    # import torch
    # from torch import nn
    # class TinyVGG(nn.Module):
    #     """Creates the TinyVGG architecture.

    #     Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    #     See the original architecture here: https://poloclub.github.io/cnn-explainer/

    #     Args:
    #     input_shape: An integer indicating number of input channels.
    #     hidden_units: An integer indicating number of hidden units between layers.
    #     output_shape: An integer indicating number of output units.
    #     """
    #     def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
    #         super().__init__()
    #         self.conv_block_1 = nn.Sequential(
    #             nn.Conv2d(in_channels=input_shape, 
    #                     out_channels=hidden_units, 
    #                     kernel_size=3, 
    #                     stride=1, 
    #                     padding=0),  
    #             nn.ReLU(),
    #             nn.Conv2d(in_channels=hidden_units, 
    #                     out_channels=hidden_units,
    #                     kernel_size=3,
    #                     stride=1,
    #                     padding=0),
    #             nn.ReLU(),
    #             nn.MaxPool2d(kernel_size=2,
    #                         stride=2)
    #         )
    #         self.conv_block_2 = nn.Sequential(
    #             nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
    #             nn.ReLU(),
    #             nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
    #             nn.ReLU(),
    #             nn.MaxPool2d(2)
    #         )
    #         self.classifier = nn.Sequential(
    #             nn.Flatten(),
    # #             nn.Linear(in_features=hidden_units*5*5,out_features=output_shape) # image shape 32, 32
    #             nn.Linear(in_features=hidden_units*53*53,out_features=output_shape) # image shape 224, 224
    #         )

    #     def forward(self, x: torch.Tensor):
    # # #         print(x.shape)
    # #         x = self.conv_block_1(x)
    # # #         print(x.shape)
    # #         x = self.conv_block_2(x)
    # # #         print(x.shape)
    # #         x = self.classifier(x)
    # #         return x
    #         return self.classifier(self.conv_block_2(self.conv_block_1(x)))



    # # Do a dummy forward pass (to test the model) 
    # model = TinyVGG(input_shape=3,
    #             hidden_units=10,
    #             output_shape=3)

    # print(f"test run: {model(torch.randn(1, 3, 224, 224))}")

    # ################################################################################

    # # setup training / testing loops
    # def train_step(model: torch.nn.Module, 
    #             dataloader: torch.utils.data.DataLoader, 
    #             loss_fn: torch.nn.Module, 
    #             optimizer: torch.optim.Optimizer,
    #             device: torch.device):
    #     # Put model in train mode
    #     model.train()
        
    #     # Setup train loss and train accuracy values
    #     loss, acc = 0, 0
        
    #     # Loop through data loader data batches
    #     for batch, (X, y) in enumerate(dataloader):
    #         # Send data to target device
    #         X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
    # #         X, y = X.to(device, non_blocking=True, memory_format=torch.channels_last), y.to(device, non_blocking=True)
    # #         X, y = X.to(device), y.to(device)

    #         # 1. Forward pass
    #         y_pred = model(X)

    #         # 2. Calculate  and accumulate loss
    #         loss = loss_fn(y_pred, y)
    #         loss += loss.item() 

    #         # 3. Optimizer zero grad
    #         optimizer.zero_grad()

    #         # 4. Loss backward
    #         loss.backward()

    #         # 5. Optimizer step
    #         optimizer.step()

    #         # Calculate and accumulate accuracy metric across all batches
    #         y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    #         acc += (y_pred_class == y).sum().item()/len(y_pred)
            
    #     # Adjust metrics to get average loss and accuracy per batch 
    #     loss = loss / len(dataloader)
    #     acc = acc / len(dataloader)
    #     return loss, acc

    # def test_step(model: torch.nn.Module, 
    #             dataloader: torch.utils.data.DataLoader, 
    #             loss_fn: torch.nn.Module,
    #             device: torch.device):
    #     # Put model in eval mode
    #     model.eval() 
        
    #     # Setup test loss and test accuracy values
    #     loss, acc = 0, 0
        
    #     # Turn on inference context manager
    #     with torch.inference_mode():
    #         # Loop through DataLoader batches
    #         for batch, (X, y) in enumerate(dataloader):
    #             # Send data to target device
    #             X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
    # #             X, y = X.to(device, non_blocking=True, memory_format=torch.channels_last), y.to(device, non_blocking=True)
    # #             X, y = X.to(device), y.to(device)
        
    #             # 1. Forward pass
    #             test_pred_logits = model(X)

    #             # 2. Calculate and accumulate loss
    #             loss = loss_fn(test_pred_logits, y)
    #             loss += loss.item()
                
    #             # Calculate and accumulate accuracy
    #             test_pred_labels = test_pred_logits.argmax(dim=1)
    #             acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
                
    #     # Adjust metrics to get average loss and accuracy per batch 
    #     loss = loss / len(dataloader)
    #     acc = acc / len(dataloader)
    #     return loss, acc

    # from tqdm.auto import tqdm

    # # 1. Take in various parameters required for training and test steps
    # def train(model: torch.nn.Module, 
    #         train_dataloader: torch.utils.data.DataLoader, 
    #         test_dataloader: torch.utils.data.DataLoader, 
    #         optimizer: torch.optim.Optimizer,
    #         loss_fn: torch.nn.Module,
    #         epochs: int,
    #         device: torch.device):
        
    #     print(f"[INFO] Training model {model.__class__.__name__} on device '{device}' for {epochs} epochs...")
        
    #     # 2. Create empty results dictionary
    #     results = {"loss": [],
    #         "acc": [],
    #         "loss": [],
    #         "acc": []
    #     }
        
    #     # 3. Loop through training and testing steps for a number of epochs
    # #     for epoch in range(epochs):
    #     for epoch in tqdm(range(epochs)):
    #         # Do eval before training (to see if there's any errors)
    #         loss, acc = test_step(model=model,
    #             dataloader=test_dataloader,
    #             loss_fn=loss_fn,
    #             device=device)
            
    #         loss, acc = train_step(model=model,
    #                                         dataloader=train_dataloader,
    #                                         loss_fn=loss_fn,
    #                                         optimizer=optimizer,
    #                                         device=device)
            
            
    #         # 4. Print out what's happening
    #         print(
    #             f"Epoch: {epoch+1} | "
    #             f"loss: {loss:.4f} | "
    #             f"acc: {acc:.4f} | "
    #             f"loss: {loss:.4f} | "
    #             f"acc: {acc:.4f}"
    #         )

    #         # 5. Update results dictionary
    #         results["loss"].append(loss)
    #         results["acc"].append(acc)
    #         results["loss"].append(loss)
    #         results["acc"].append(acc)

    #     # 6. Return the filled results at the end of the epochs
    #     return results

    # ################################################################################

    # # train the model on CPU and GPU (MPS device)
    # # Set random seed
    # torch.manual_seed(42)

    # # Create device list
    # devices = ["cpu", "mps"]

    # for device in devices:

    #     # Recreate an instance of TinyVGG
    #     model = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
    #                     hidden_units=10, 
    #                     output_shape=len(train_data.classes)).to(device)

    #     # Setup loss function and optimizer
    #     loss_fn = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    #     # Start the timer
    #     from timeit import default_timer as timer 
    #     start_time = timer()

    #     # Train model
    #     model_results = train(model=model, 
    #                         train_dataloader=train_dataloader,
    #                         test_dataloader=test_dataloader,
    #                         optimizer=optimizer,
    #                         loss_fn=loss_fn, 
    #                         epochs=NUM_EPOCHS,
    #                         device=device)

    #     # End the timer
    #     end_time = timer()

    #     # Print out timer and results
    #     total_train_time = print_train_time(start=start_time,
    #                                         end=end_time,
    #                                         device=device,
    #                                         machine=MACHINE)
        
    #     # Create results dict
    #     results = {
    #     "machine": MACHINE,
    #     "device": device,
    #     "dataset_name": DATASET_NAME,
    #     "epochs": NUM_EPOCHS,
    #     "batch_size": BATCH_SIZE,
    #     "image_size": IMAGE_SIZE[0],
    #     "num_train_samples": len(train_data),
    #     "num_test_samples": len(test_data),
    #     "total_train_time": round(total_train_time, 3),
    #     "time_per_epoch": round(total_train_time/NUM_EPOCHS, 3),
    #     "model": model.__class__.__name__,
    #     "test_accuracy": model_results["acc"][-1]
    #     }
        
    #     results_df = pd.DataFrame(results, index=[0])
        
    #     # Write CSV to file
    #     import os
    #     if not os.path.exists("results/"):
    #         os.makedirs("results/")

    #     results_df.to_csv(f"results/{MACHINE.lower().replace(' ', '_')}_{device}_{DATASET_NAME}_image_size.csv", 
    #                     index=False)

    # ################################################################################

    # # inspect the results
    # # Get CSV paths from results
    # from pathlib import Path
    # results_paths = list(Path("results").glob("*.csv"))

    # df_list = []
    # for path in results_paths:
    #     df_list.append(pd.read_csv(path))
    # results_df = pd.concat(df_list).reset_index(drop=True)
    # print(results_df)

    # # Get names of devices
    # machine_and_device_list = [row[1][0] + " (" + row[1][1] + ")" for row in results_df[["machine", "device"]].iterrows()]

    # # Plot and save figure
    # plt.figure(figsize=(10, 7))
    # plt.style.use('fivethirtyeight')
    # plt.bar(machine_and_device_list, height=results_df.time_per_epoch)
    # plt.title(f"PyTorch TinyVGG Training on CIFAR10 with batch size {BATCH_SIZE} and image size {IMAGE_SIZE}", size=16)
    # plt.xlabel("Machine (device)", size=14)
    # plt.ylabel("Seconds per epoch (lower is better)", size=14);
    # save_path = f"results/{model.__class__.__name__}_{DATASET_NAME}_benchmark_with_batch_size_{BATCH_SIZE}_image_size_{IMAGE_SIZE[0]}.png"
    # print(f"Saving figure to '{save_path}'")
    # plt.savefig(save_path)

