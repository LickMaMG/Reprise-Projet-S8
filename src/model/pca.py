from sklearn.decomposition import IncrementalPCA
from keras.utils import Progbar

from metrics import MSE
from callbacks import SaveDenoised


class CustomPCA:
    name = "pca"
    
    def __init__(self, n_components: int) -> None:
        self.pca = IncrementalPCA(n_components=n_components)
        self.metrics = [MSE()]
        self.callbacks = []

    def fit(self, train_gen, callbacks = None, metrics = None):
        if metrics is not None: self.metrics.extend(metrics)
        if callbacks is not None: self.callbacks.extend(callbacks)

        batch_size = train_gen.batch_size

        metrics_val      = [[m.name, []] for m in self.metrics]
        metrics_mean_val = [[m.name, 0] for m in self.metrics]

        self.prog = Progbar(len(train_gen))
        for i, (noised, labels) in enumerate(train_gen):
            self.pca.partial_fit(noised.reshape(batch_size, -1))

            reduced = self.pca.transform(noised.reshape(batch_size, -1))
            denoised = self.pca.inverse_transform(reduced).reshape(batch_size, *noised[0].shape[:2])
            y_true = labels.reshape(batch_size, -1)
            y_pred = denoised.reshape(batch_size, -1)

            y_true = labels.reshape(batch_size, -1)
            y_pred = denoised.reshape(batch_size, -1)

            for j, m in enumerate(self.metrics):
                metrics_val[j][1].append(m(y_true, y_pred))
                metrics_mean_val[j][1] = sum(metrics_val[j][1])/len(metrics_val[j][1])
            
            self.prog.update(i+1, values=metrics_mean_val)
        
    def evaluate(self, logdir: str, generator):
        callback = SaveDenoised(logdir=logdir, generator=generator)
        
        metrics_val      = [[m.name, []] for m in self.metrics]
        metrics_mean_val = [[m.name, 0] for m in self.metrics]

        batch_size = generator.batch_size
        prog = Progbar(len(generator))
        for i, (noised , labels) in enumerate(generator):
            reduced = self.pca.transform(noised.reshape(batch_size, -1))
            denoised = self.pca.inverse_transform(reduced).reshape(batch_size, *noised[0].shape[:2])

            y_true = labels.reshape(batch_size, -1)
            y_pred = denoised.reshape(batch_size, -1)

            y_true = labels.reshape(batch_size, -1)
            y_pred = denoised.reshape(batch_size, -1)

            callback(bacth_num=i, noised=noised, labels=labels)

            for j, m in enumerate(self.metrics):
                metrics_val[j][1].append(m(y_true, y_pred))
                metrics_mean_val[j][1] = sum(metrics_val[j][1])/len(metrics_val[j][1])
            
            prog.update(i+1, values=metrics_mean_val)
        
        for name, val in metrics_mean_val:
            print(name.ljust(10) + " : %.2f" % val)



