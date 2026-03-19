# Privacy-Preserving Federated Learning for Financial Transaction Analysis

This project implements a decentralized, privacy-preserving machine learning system specifically designed for detecting fraudulent financial transactions across multiple banking institutions. By leveraging Federated Learning (FL), participating banks can collaboratively train a robust fraud detection model without ever sharing their raw, sensitive customer transaction data.

## 🌟 Key Features

- **Federated Learning Architecture**: Utilizes [Flower](https://flower.ai/) (`flwr`) to manage federated training across multiple simulated bank clients.
- **Privacy-Preserving**: Raw transaction data never leaves the client's local environment. Only model weights are shared with the central server to learn global patterns.
- **PyTorch Neural Network**: Implements a deep learning model (`SimpleNN`) capable of processing both categorical and numerical financial data.
- **Automated Training Pipeline**: Includes a single-click scripting approach (`app.py`) to automatically orchestrate the server and all client nodes.
- **Interactive Web Interface**: A beautifully designed Flask-based web application (`main.py`) where users can input transaction details and receive real-time fraud predictions based on the globally trained federated model.

---

## 🏗️ Project Structure

```text
📁 federated-airfare (2) - Copy
├── app.py                  # Orchestration script to run the FL server and clients sequentially.
├── server.py               # FL server utilizing FedAvg strategy to aggregate model weights.
├── client.py               # FL client representing a simulated bank holding local transaction data.
├── main.py                 # Flask web application for interactive real-time predictions.
├── model.py                # PyTorch neural network architecture details (`SimpleNN`).
├── preproces.py            # Data preprocessing utilities (e.g. label encoding).
├── splitdataset.py         # Utility to split the central dataset into disparate banks (simulating horizontal FL).
├── label_encoders.pkl      # Serialized encoders to standardize categorical inputs for inference.
├── federated_model.pth     # The resulting trained global model weights.
├── requirements.txt        # Python package dependencies.
└── templates/ & static/    # HTML/CSS assets for the Flask web application.
```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed on your system.

Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

### 2. Federated Training

To simulate the federated learning process, you can launch the orchestration script. This will spin up the server and 4 bank clients, run the training rounds, and output the aggregated global model (`federated_model.pth`).

```bash
python app.py
```
*Note: Ensure that the client datasets (`bank_1.csv`, `bank_2.csv`, etc.) are available in the directory before starting.*

### 3. Running the Web Interface

Once the global model is trained and saved, you can launch the web-based inference interface:

```bash
python main.py
```

Navigate to `http://127.0.0.1:5000/` in your browser. From here, you can input transaction characteristics (e.g., amount, device type, merchant category) to get a real-time Fraud/Not Fraud prediction.

---

## 🛡️ How Federated Learning Enhances Privacy

1. **Local Training**: Each bank (Client) trains a local copy of the neural network using its private transaction records.
2. **Weight Aggregation**: Once trained, only the mathematical weight updates are sent to the central Server.
3. **Global Update**: The Server uses Federated Averaging (`FedAvg`) to aggregate the learnings from all banks into a single Global Model.
4. **Distribution**: The improved Global Model is sent back to the banks for the next round of training or used directly for inference.

This ensures **zero data leakage** among competing financial institutions while still benefiting from a shared intelligence network to combat fraud!
