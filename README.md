![Build Status](https://github.com/ManivelSethu/Session5/actions/workflows/ml-pipeline.yml/badge.svg)
# Session5
Session5 - ML OPs CI/CD and a MNIST based model

**Model Architecture:**
MNISTNet(
  (conv1): Conv2d(1, 4, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=200, out_features=32, bias=True)
  (fc2): Linear(in_features=32, out_features=10, bias=True)
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)

**Model Parameter Details:**
Total parameters: 7,098
Trainable parameters: 7,098

**Added 3 new test cases**
### 1. Test Model Robustness
- **Function:** `test_model_robustness`
- **Description:** Evaluates the model's performance against noisy inputs by adding Gaussian noise to test images and checking prediction consistency.
- **Purpose:** Ensures the model is robust to variations in input data.

### 2. Test Model Batch Inference
- **Function:** `test_model_batch_inference`
- **Description:** Assesses the model's efficiency and performance with different batch sizes, measuring inference time and output shape.
- **Purpose:** Validates the model's scalability and ability to handle varying input sizes.

### 3. Test Model Save and Load
- **Function:** `test_model_save_load`
- **Description:** Verifies the model's ability to be saved and loaded correctly using both state dictionary and TorchScript methods, ensuring output consistency.
- **Purpose:** Ensures model serialization and deserialization for deployment and versioning.

