import numpy as np
from scipy.special import expit
import math
from mnist import load_dataset, draw

class Network:
    
    INPUT_NODES = 784
    OUTPUT_NODES = 10
    
    def __init__(self, hidden_nodes, learning_rate, weights=None):
        self.HIDDEN_NODES = hidden_nodes
        self.learning_rate = learning_rate
        if weights is None:
            self.set_random_weights()
        else:
            self.weights_input_hidden = weights[0]
            self.weights_hidden_output = weights[1]
        
    def set_random_weights(self):
        self.weights_input_hidden = np.random.normal(0, 1 / math.sqrt(Network.INPUT_NODES), size=(self.HIDDEN_NODES, Network.INPUT_NODES))
        self.weights_hidden_output = np.random.normal(0, 1 / math.sqrt(self.HIDDEN_NODES), size=(Network.OUTPUT_NODES, self.HIDDEN_NODES))
                    
    def train_one(self, image, digit):
        input_to_hidden_layer = np.dot(self.weights_input_hidden, image)
        output_of_hidden_layer = expit(input_to_hidden_layer)
        input_to_output_layer = np.dot(self.weights_hidden_output, output_of_hidden_layer)
        output_of_output_layer = expit(input_to_output_layer)
        should_values = np.zeros(10) + 0.01
        should_values[digit] = 0.99
        error_output = should_values - output_of_output_layer
        #error_hidden = np.dot(np.transpose(self.weights_hidden_output), error_output) # This is easier and works as well.
        error_hidden = np.dot(np.transpose(self.weights_hidden_output) / np.sum(np.abs(self.weights_hidden_output), axis=1), error_output) * (self.HIDDEN_NODES / 10)
        delta_w_output = np.outer((should_values - output_of_output_layer) * output_of_output_layer * (1 - output_of_output_layer), output_of_hidden_layer)
        delta_w_hidden = np.outer(error_hidden * output_of_hidden_layer * (1 - output_of_hidden_layer), image)
        self.weights_input_hidden += self.learning_rate  * delta_w_hidden
        self.weights_hidden_output += self.learning_rate * delta_w_output
      
    def test_one(self, input_nodes):
        input_to_hidden_layer = np.dot(self.weights_input_hidden, input_nodes)
        output_of_hidden_layer = expit(input_to_hidden_layer)
        input_to_output_layer = np.dot(self.weights_hidden_output, output_of_hidden_layer)
        output_of_output_layer = expit(input_to_output_layer)
        return output_of_output_layer
                 
    def train(self, epochs, train_imgs, train_labels):
        for epoch in range(epochs):
            for i in range(60000):
                network.train_one(train_imgs[i], train_labels[i])
                if (i % 1000 == 0):
                    print(f"Epoch {epoch+1} from {epochs}: {round(i / 600, 2)}%")
    
    def test(self, test_imgs, test_labels):
        right = 0
        wrong = 0
        wrong_test_cases = []     
        for i in range(10000):
            output = network.test_one(test_imgs[i])
            prediction = self.get_prediction(output)
            if prediction == test_labels[i]:
                right += 1
            else:
                wrong += 1
                wrong_test_cases.append(i)     
            if (i % 1000 == 0):
                print(f"Testing... {round(i / 100, 2)}%")          
        print(f"right: {right}, wrong: {wrong}")
        print(f"Accuracy: {round(right / 100, 2)}%")
        return wrong_test_cases
    
    def get_prediction(self, output):
        max_value = 0
        prediction = 0
        for j in range(10):
            if output[j] > max_value:
                max_value = output[j]
                prediction = j
        return prediction
        
    def print_prediction(self, image):
        print()
        output = self.test_one(image)
        for i, v in enumerate(output):
            print(f"{i}: {round(v, 3)}")
        prediction = self.get_prediction(output)
        print(f"Prediction {prediction}")
        
    def draw_and_predict(self, test_imgs, index):
        network.print_prediction(test_imgs[index])
        draw(test_imgs[index])  
        
    def get_weights(self):
        return (self.weights_input_hidden.copy(), self.weights_hidden_output.copy())
    
    def save_weights(self, hidden_filename, output_filename):
        np.save(hidden_filename, self.weights_input_hidden.copy())
        np.save(output_filename, self.weights_hidden_output.copy())
        

def get_pretrained_network():
    weights_input_hidden = np.load("../res/pretrained_weights/hidden_weights.npy")
    weights_hidden_output = np.load("../res/pretrained_weights/output_weights.npy")
    network = Network(200, 0.1, (weights_input_hidden, weights_hidden_output))
    return network


if __name__ == "__main__":
    
    train_imgs, train_labels, test_imgs, test_labels = load_dataset()
    
    #network = get_pretrained_network()
    network = Network(200, 0.1)
    network.train(2, train_imgs, train_labels)
    
    wrong_test_cases = network.test(test_imgs, test_labels)

    #network.draw_and_predict(test_imgs, 4522)
      
