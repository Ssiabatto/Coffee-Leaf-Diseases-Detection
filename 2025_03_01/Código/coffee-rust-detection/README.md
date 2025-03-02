# Coffee Rust Detection Project

This project aims to build an AI model for detecting coffee rust on leaves using TensorFlow and MobileNetV2. The model will be trained on a dataset consisting of images of both healthy and infected coffee leaves.

## Project Structure

```
coffee-rust-detection
├── data
│   ├── healthy          # Contains images of healthy coffee leaves
│   └── infected         # Contains images of infected coffee leaves
├── notebooks
│   └── data_preprocessing.ipynb  # Jupyter notebook for data preprocessing
├── src
│   ├── ai.py            # Main code for building the AI model
│   ├── train.py         # Code for training the AI model
│   ├── evaluate.py      # Code for evaluating the trained model
│   └── utils
│       └── data_loader.py  # Utility functions for loading and preprocessing data
├── requirements.txt      # Lists project dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <https://github.com/Ssiabatto/Coffee-Leaf-Diseases-Detection.git>
   cd coffee-rust-detection
   ```

2. **Install the required dependencies**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.
   ```
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**:
   - Place images of healthy coffee leaves in the `data/healthy` directory.
   - Place images of infected coffee leaves in the `data/infected` directory.

## Usage Guidelines

- **Data Preprocessing**: Use the Jupyter notebook located in notebooks/data_preprocessing.ipynb to preprocess the dataset. This includes loading images, resizing, and augmenting the dataset for training.

- **Training the Model**: RRun the train.py script to train the AI model. Ensure that the dataset is properly set up before running this script.

- **Evaluating the Model**: After training, use the evaluate.py script to evaluate the model's performance on a validation or test dataset. The script will display 10 images with their predictions and actual labels, followed by 5 individual images with their predictions and actual labels.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.