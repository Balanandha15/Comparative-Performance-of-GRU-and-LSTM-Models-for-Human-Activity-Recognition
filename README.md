# Comparative-Performance-of-GRU-and-LSTM-Models-for-Human-Activity-Recognition


This project provides a comparative analysis of two recurrent neural network (RNN) architectures—Gated Recurrent Units (GRU) and Long Short-Term Memory (LSTM)—for Human Activity Recognition (HAR) using sensor data. The study utilizes the UCI HAR Dataset, which consists of accelerometer and gyroscope data collected from wearable devices. By comparing the performance of GRU and LSTM, this project aims to highlight their capabilities in accurately classifying various human activities.

 Dataset
 
UCI Human Activity Recognition (HAR) Dataset: The dataset contains time-series data from accelerometers and gyroscopes. It includes labeled activities such as walking, sitting, and standing, allowing for the evaluation of model performance in multi-class classification.

Dataset Download from : https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

 Project Structure

  Data Preprocessing: The `DataLoader` class loads, scales, reshapes, and one-hot encodes the HAR dataset, preparing it for input into the RNN models.
  Model Architecture: The `RNNModel` class defines both GRU and LSTM models, each comprising two recurrent layers and dropout for regularization.
  Training and Evaluation: Each model is trained on the dataset, with metrics such as accuracy and loss plotted over epochs. A confusion matrix is used to analyze the classification accuracy of each model on individual activities.

Key Files

- `Comparative Performance of GRU and LSTM Models for Human Activity Recognition.ipynb`: The main notebook that contains code for data loading, model definition, training, evaluation, and visualization of results.

Abstract:

  Human Activity Recognition (HAR) is a crucial area in the field of wearable computing and healthcare, enabling the detection of daily activities using sensor data. This research focuses on the comparative performance of two recurrent neural network (RNN) architectures, Gated Recurrent Units (GRU) and Long Short-Term Memory (LSTM) networks, for human activity recognition. Using the UCI HAR dataset, which consists of sensor data from accelerometers and gyroscopes, we evaluate the performance of both models in terms of accuracy, training history, and confusion matrix. The results show that both GRU and LSTM models achieve high accuracy, with GRU slightly outperforming LSTM in terms of classification accuracy. This study highlights the potential of RNNs, particularly GRU and LSTM, in real-time activity monitoring and lays the foundation for further advancements in human activity recognition systems.

Objective:

  The objective of this research is to conduct a detailed comparison between Gated Recurrent Units (GRU) and Long Short-Term Memory (LSTM) networks for Human Activity Recognition (HAR) using the UCI HAR dataset, which contains sensor data from accelerometers and gyroscopes. The study aims to evaluate the performance of both models in terms of classification accuracy, loss, and generalization ability across training and test datasets. Additionally, it seeks to analyze the models’ training dynamics, including convergence rates and overfitting tendencies, and to generate confusion matrices to assess their ability to correctly classify different human activities. By comparing these two popular Recurrent Neural Network (RNN) architectures, the research will provide insights into their relative strengths and limitations for HAR tasks, with the goal of improving real-time activity recognition and contributing to the development of more robust, efficient, and scalable activity recognition systems.

Model used:

1) GRU

The GRU Model Architecture is a type of RNN that addresses the vanishing gradient problem present in traditional RNNs while being computationally more efficient than LSTMs. The GRU model consists of two recurrent layers. The first layer contains 64 units, and it uses the `return sequences=True` parameter to pass the full sequence of outputs to the next layer. The second layer has 32 units. This structure allows the model to learn complex patterns from the sequential data. To prevent overfitting and improve generalization to unseen data, Dropout layers are added after each recurrent layer, with a dropout rate of 0.5. The output layer is a Dense layer with 6 neurons, each corresponding to one of the six activity classes. It uses a softmax activation function, which produces a probability distribution over the classes, with the highest probability indicating the predicted activity.

2) LSTM

The LSTM Model Architecture is another advanced RNN variant designed to capture long-term dependencies in sequential data. It utilizes memory cells and gates to control the flow of information, which is particularly useful in tasks like activity recognition, where long sequences of events (e.g., sensor data over time) need to be processed. Similar to the GRU model, the LSTM model also has two recurrent layers, with the first layer containing 64 units and `return sequences =True` to ensure that the sequence is passed to the next layer. The second layer consists of 32 units. Dropout layers, with a rate of 0.5, are included after each LSTM layer to help prevent overfitting and improve the model's generalization to new data. The LSTM model also has a Dense output layer with 6 neurons, representing the six different activity classes, and uses a softmax activation function to produce a probability distribution for each class.

Conclusion:

In this research, both GRU and LSTM models were implemented and evaluated for the task of Human Activity Recognition using a public dataset. The GRU model demonstrated a test accuracy of 95.08\% with a loss of 0.1552, outperforming the LSTM model, which achieved an accuracy of 94.03\% and a loss of 0.1713. While both models performed effectively, the GRU model showed a slight edge due to its simpler architecture and lower computational demands, leading to better generalization on the test data. This finding suggests that for tasks involving time-series data with manageable long-term dependencies, GRU can offer a preferable balance between accuracy and computational efficiency compared to LSTM. Thus, GRU models can be recommended as a more efficient alternative to LSTMs in similar applications, especially when computational resources are limited or when achieving marginally better accuracy and loss values is desirable. This conclusion highlights the potential of GRU for efficient and accurate human activity recognition in real-world applications
