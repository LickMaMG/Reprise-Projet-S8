from sklearn.decomposition import IncrementalPCA
from keras.utils import Progbar

from metrics import MSE

class CustomPCA:
    name = "pca"
    
    def __init__(self, n_components: int) -> None:
        self.pca = IncrementalPCA(n_components=n_components)
        self.metrics = [MSE()]
        self.callbacks = []

    def fit(self, train_gen, callbacks = None, metrics = None):
        if metrics is not None: self.metrics.extend(metrics)
        if callbacks is not None: self.callbacks.extend(callbacks)

        metrics_val      = [[m.name, []] for m in self.metrics]
        metrics_mean_val = [[m.name, 0] for m in self.metrics]

        self.prog = Progbar(len(train_gen))
        for i, (batch, labels) in enumerate(train_gen):
            self.pca.partial_fit(batch.reshape(train_gen.batch_size, -1))

            reduced = self.pca.transform(batch.reshape(train_gen.batch_size, -1))
            denoised = self.pca.inverse_transform(reduced).reshape(train_gen.batch_size, *batch[0].shape[:2])
            y_true = labels.reshape(train_gen.batch_size, -1)
            y_pred = denoised.reshape(train_gen.batch_size, -1)

            y_true = labels.reshape(train_gen.batch_size, -1)
            y_pred = denoised.reshape(train_gen.batch_size, -1)

            for j, m in enumerate(self.metrics):
                metrics_val[j][1].append(m(y_true, y_pred))
                metrics_mean_val[j][1] = sum(metrics_val[j][1])/len(metrics_val[j][1])
            
            self.prog.update(i+1, values=metrics_mean_val)
        
    def evaluate(self, generator):
        metrics_val = [[m.name, 0] for m in self.metrics]
        for i, (batch , labels) in enumerate(generator):
            reduced = self.pca.transform(batch.reshape(generator.batch_size, -1))
            denoised = self.pca.inverse_transform(reduced).reshape(generator.batch_size, *batch[0].shape[:2])
            y_true = labels.reshape(generator.batch_size, -1)
            y_pred = denoised.reshape(generator.batch_size, -1)

            y_true = labels.reshape(generator.batch_size, -1)
            y_pred = denoised.reshape(generator.batch_size, -1)

            for i, m in enumerate(self.metrics):
                metrics_val[i][1] += m(y_true, y_pred)
        
        metrics_val = [(name, val/len(generator)) for name,val in metrics_val]
            
        # metrics_val = []
        for name, val in metrics_val:
            print(name.ljust(10) + " : %.2f" % val)



