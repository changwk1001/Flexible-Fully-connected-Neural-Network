import numpy as np
import sys
class neuralnetwork():
    def __init__(self,args):
        
        self.learning_rate = args["learning_rate"]
        self.iteration = args["training_iteration"]
        self.iteration = 10
        self.lossfunction = args["lossfunction"]
        self.activationfunction = args["activation_function"]
        self.random = np.random.RandomState(args["seed"])
        self.numberoflayer = args["numberoflayer"]
        self.sequenceoflayer = args["sequenceoflayer"]
        self.regulation = args["regulation"]
        self.shuffle = True
        self.batch_size = args["batch_size"]
        self.regulation_strength = args["regulation_strength"]
        
    def _parameter_initialize(self):
        self.w = list()
        self.b = list()
        for i in range(self.numberoflayer - 1):
            self.w.append(np.random.normal(loc=0,scale=0.1,size=(self.sequenceoflayer[i],self.sequenceoflayer[i+1])))
            self.b.append(np.random.normal(loc=0,scale=0.1,size=self.sequenceoflayer[i+1]))
            
    def _activation(self,x,derivative):
        if self.activationfunction == 'logistic' and derivative == 'off':
            return 1 / (1 + np.exp(-np.clip(x,-250,250)))
        elif self.activationfunction == 'logistic' and derivative == 'on':
            return self._activation(x,'off') * (1 - self._activation(x,'off'))
        elif self.activationfunction == 'RELU' and derivative == 'off':
            return x
        elif self.activationfunction == 'RELU' and derivative == 'on':
            return np.ones((x.shape[0],x.shape[1]))
        
    def predict(self,x):
        output = self._forwardpropogation(x)
        y_pred = np.argmax(output,axis=1) #z_out is the correct since there is clip in calculating the cost
        return y_pred
    
    def _forwardpropogation(self,x):
        self.node_value = list()
        self.node_value.append(x)
        for i in range(self.numberoflayer - 1):
            self.node_value.append(self._activation(self.node_value[i].dot(self.w[i] + self.b[i]) ,derivative='off'))
        return self.node_value[-1]
    
    def _lossfunction(self,y_enc,output,derivative):
        
        # Derivative off and square
        if self.lossfunction == 'square' and derivative == 'off' and self.regulation == 'none':
            return np.sum(np.dot(y_enc-output, (y_enc-output).T),axis=0)
        elif self.lossfunction == 'square' and derivative == 'off' and self.regulation == 'l2':
            l2 = 0
            for i in range(self.numberoflayer - 1):
               l2 += np.sum(np.dot(self.w[i],self.w[i].T)) 
            return np.sum(np.dot(y_enc-output, (y_enc-output).T)) + l2
        
        # Derivative off and logistic
        elif self.lossfunction == 'logistic' and derivative == 'off' and self.regulation == 'none':
            term1 = -y_enc * np.log(output)
            term2 = (1. - y_enc) * np.log(1. - output)
            cost = np.sum(term1-term2)
            return cost
        elif self.lossfunction == 'logistic' and derivative == 'off' and self.regulation == 'l2':
            l2 = 0
            for i in range(self.numberoflayer - 1):
               l2 += np.sum(np.dot(self.w[i],self.w[i].T))
            term1 = -y_enc * np.log(output)
            term2 = (1. - y_enc) * np.log(1. - output)
            cost = np.sum(term1-term2)
            return cost + l2
        
        # Derivative on and logistic
        elif self.lossfunction == 'logistic' and derivative == 'on' \
            and self.activationfunction == 'logistic':
            return (-y_enc + output)
        elif self.lossfunction == 'square' and derivative == 'on' and self.activationfunction == 'RELU':
            return -(y_enc-output)
        elif self.lossfunction == 'square' and derivative == 'on' and self.activationfunction == 'logistic':
            return -(y_enc-output) * self._activation(output,'on')
        
    def _onehot(self,y,n_classes):
        onehot = np.zeros((n_classes,y.shape[0]))
        for idx,val in enumerate(y.astype(int)):
            onehot[val,idx] = 1
        return onehot.T
    
    def fit(self,x_train,y_train,x_valid,y_valid):
        n_output = np.unique(y_train).shape[0]
        y_train_enc = self._onehot(y_train,n_output)       
        epoch_strlen = len(str(self.iteration))
        self.eval_ = {'train_acc':[],'valid_acc':[]}       
        self._parameter_initialize()
        self.round = 0
        
        for q in range(self.iteration):
            indices = np.arange(x_train.shape[0])
            
            # Shuffling the data to prevent jointly train the same bunch of picture
            if self.shuffle: 
                self.random.shuffle(indices)
                
            for start_idx in range(0,indices.shape[0]-self.batch_size+1,self.batch_size):
                batch_idx = indices[start_idx:start_idx+self.batch_size]
                output = self._forwardpropogation(x_train[batch_idx])
                
                # Backpropogation
                g = self._lossfunction(y_train_enc[batch_idx],output,'on')
                self.b_changelist = list()
                self.w_changelist = list()
                for i in range(self.numberoflayer-1,0,-1):
                    self.round += 1
                    if self.regulation == 'l2':
                        self.b_changelist.insert(0, np.sum(g,axis=0) + self.regulation_strength * 2 * self.b[i-1])
                        self.w_changelist.insert(0, self.node_value[i-1].T.dot(g) + self.regulation_strength * 2 * self.w[i-1])
                        g = g.dot(self.w[i-1].T)
                        g = g * self._activation(self.node_value[i-1],'on')
                    elif self.regulation == 'none':
                        self.b_changelist.insert(0, np.sum(g,axis=0))
                        self.w_changelist.insert(0, self.node_value[i-1].T.dot(g))
                        g = g.dot(self.w[i-1].T)
                        g = g * self._activation(self.node_value[i-1],'on')
                for i in range(self.numberoflayer-1):
                    self.w[i] -= self.w_changelist[i] * self.learning_rate
                    self.b[i] -= self.b_changelist[i] * self.learning_rate
                
                # Evulation
                if self.round % 1000 == 0:    
                    output = self._forwardpropogation(x_train[batch_idx])
                    y_train_pred = self.predict(x_train[batch_idx])
                    y_valid_pred = self.predict(x_valid)
                    train_acc = ((np.sum(y_train[batch_idx]==y_train_pred)).astype(np.float) / self.batch_size)
                    valid_acc = ((np.sum(y_valid==y_valid_pred)).astype(np.float) / x_valid.shape[0])
                    sys.stderr.write('\r%0*d/%d | train/valid acc: %.2f%%/%.2f%%' %(epoch_strlen,
                                                        q+1,self.iteration,train_acc*100,valid_acc*100))
                    self.eval_['train_acc'].append(train_acc)
                    self.eval_['valid_acc'].append(valid_acc)
                    
        return self
