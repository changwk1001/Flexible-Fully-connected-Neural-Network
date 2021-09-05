import argparse

def get_parser():

    parser = argparse.ArgumentParser("Learner",
                                     description="Learning trainer.")
    # Model related:
    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float,
                        help="Learning rate of the model")
    
    parser.add_argument("-it", "--training_iteration", default=10, type=int,
                        help="Training iteration of your model")
    
    parser.add_argument("-loss", "--lossfunction", default="square", type=str,
                        choices=["square", "logistic"], help="You can deside your loss function here")
    
    parser.add_argument("-a", "--activation_function", default="RELU", type=str,
                        choices=["RELU", "logistic"], help="You can choice the activation function for your layer")
    
    parser.add_argument("-s", "--seed", default=1, type=int,
                        help="This is the seed for your initialization of the model weight and bias")
    
    parser.add_argument("-b", "--batch_size", default=1, type=int,
                        help="You can desice how many picture to train jointly")
    
    parser.add_argument("-nl", "--numberoflayer", default=4, type=int,
                        help="How many layers you want for your model (input and output included)?")
    
    parser.add_argument("-sl", "--sequenceoflayer", default=[784,40,20,10], type=list,
                        help="You can define the width of your model corresponding to each layer!!")
    
    parser.add_argument("-r", "--regulation", default="none", type=str,
                        choices=["none", "l2"], help="help prevent the overfitting")
    
    parser.add_argument("-rs", "--regulation_strength", default=0.01, type=float,
                        help="Deside the strength of your regulation")
    
    return parser
