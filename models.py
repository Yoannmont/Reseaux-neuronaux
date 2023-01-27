# 2235148 : Yoann MONTEIRO

import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        w = self.get_weights()
        return nn.DotProduct(w,x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        scalar = nn.as_scalar(self.run(x))
        if scalar >=0:
            return 1
        else:
            return -1

        

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        well_classified = False
        while not well_classified :
            well_classified = True
            for x,y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    #Si au moins un élément est mal classé, on réitère sur le jeu de données
                    self.w.update(x,nn.as_scalar(y))     
                    well_classified = False



class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here

        #Paramètres d'apprentissage
        self.batch_size = 20
        self.hidden_layer_dim1 = 50
        self.hidden_layer_dim2 = 10
        self.alpha = 0.05

        #Paramètres de la première opération linéaire
        self.w1 = nn.Parameter(1,self.hidden_layer_dim1)
        self.b1 = nn.Parameter(1,self.hidden_layer_dim1)

        #Paramètres de la seconde opération linéaire
        self.w2 = nn.Parameter(self.hidden_layer_dim1,self.hidden_layer_dim2)
        self.b2 = nn.Parameter(1,self.hidden_layer_dim2)

        #Paramètres de la troisième opération linéaire
        self.w3 = nn.Parameter(self.hidden_layer_dim2,1)
        self.b3 = nn.Parameter(1,1)

        

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        #On utilise un réseau à 3 couches et 2 activations non linéaires
        layer1 = nn.AddBias(nn.Linear(x,self.w1),self.b1)
        hidden_layer1 = nn.ReLU(layer1)
        layer2 = nn.AddBias(nn.Linear(hidden_layer1,self.w2), self.b2)
        hidden_layer2 = nn.ReLU(layer2)
        output = nn.AddBias(nn.Linear(hidden_layer2,self.w3), self.b3)
        return output


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y,y)
        

    def train(self, dataset):
        """
        Trains the model.
        """
        #Informations pour comparer les performances obtenues avec les paramètres d'apprentissage
        #processed = 0
        #loops = 0

        for x,y in dataset.iterate_forever(self.batch_size):
            grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3 = nn.gradients(self.get_loss(x,y),[self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
            self.w1.update(grad_w1,-self.alpha)
            self.b1.update(grad_b1,-self.alpha)
            self.w2.update(grad_w2,-self.alpha)
            self.b2.update(grad_b2,-self.alpha)
            self.w3.update(grad_w3, -self.alpha)
            self.b3.update(grad_b3, -self.alpha)
            #processed += self.batch_size
            #loops += 1 
            current_loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x),nn.Constant(dataset.y)))
            #print("Nombre d'operations / boucles : " + str(processed)+ "/" +str(loops))
            if (current_loss < 0.02):
                break
        


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        #Paramètres d'apprentissage
        self.batch_size = 240
        self.hidden_layer_dim1 = 100
        self.hidden_layer_dim2 = 100
        self.alpha = 0.425

        #Paramètres de la première opération linéaire
        self.w1 = nn.Parameter(784,self.hidden_layer_dim1)
        self.b1 = nn.Parameter(1,self.hidden_layer_dim1)

        #Paramètres de la seconde opération linéaire
        self.w2 = nn.Parameter(self.hidden_layer_dim1,self.hidden_layer_dim2)
        self.b2 = nn.Parameter(1,self.hidden_layer_dim2)

        #Paramètres de la troisième opération linéaire
        self.w3 = nn.Parameter(self.hidden_layer_dim2,10)
        self.b3 = nn.Parameter(1,10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        #On utilise un réseau de neurones à 4 couches et 3 couches cachées
        layer1 = nn.AddBias(nn.Linear(x, self.w1),self.b1)
        hidden_layer1 = nn.ReLU(layer1)
        layer2 = nn.AddBias(nn.Linear(hidden_layer1, self.w2),self.b2)
        hidden_layer2 = nn.ReLU(layer2)
        output = nn.AddBias(nn.Linear(hidden_layer2, self.w3),self.b3)

        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        predicted_y = self.run(x)
        return nn.SoftmaxLoss(predicted_y,y)

    def train(self, dataset):
        """
        Trains the model.
        """
        for x,y in dataset.iterate_forever(self.batch_size):
            grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3 = nn.gradients(self.get_loss(x,y),[self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
            self.w1.update(grad_w1,-self.alpha)
            self.b1.update(grad_b1,-self.alpha)
            self.w2.update(grad_w2,-self.alpha)
            self.b2.update(grad_b2,-self.alpha)
            self.w3.update(grad_w3,-self.alpha)
            self.b3.update(grad_b3,-self.alpha)
            #On arrête l'apprentissage lorsqu'on a un score supérieur à 97,25 % sur le jeu de donnés de validation
            if (dataset.get_validation_accuracy() > 0.9725):
                break
