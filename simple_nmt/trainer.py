import numpy as np

import torch
from torch import optim
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from simple_nmt.utils import get_grad_norm, get_parameter_norm


VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MaximumLikelihoodEstimationEngine(Engine):

    def __init__(self, func, model, crit, optimizer, lr_scheduler, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config

        super().__init__(func) # SingleTrainer class will exploit this line.
                               # For train_engine, func = MaximumLikelihoodEstimationEngine.train
                               # For validation_engine, func = MaximumLikelihoodEstimationEngine.validate
                               # You can apply this trick to other cases, so get used to it.

        self.best_loss = np.inf
        self.scaler = GradScaler()

    @staticmethod
    #@profile
    def train(engine, mini_batch):
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train()
        if engine.state.iteration % engine.config.iteration_per_update == 1 or \
            engine.config.iteration_per_update == 1:
            if engine.state.iteration > 1:
                engine.optimizer.zero_grad()

        device = next(engine.model.parameters()).device
        mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1]) # tensors and lenths
        mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1]) # tensors and lenths 
                                                                           # the first column of tgt: <BOS>
                                                                           # the last column of src: <EOS>

        # Raw target variable has both BOS and EOS token. 
        # The output of sequence-to-sequence does not have BOS token. 
        # Thus, remove BOS token for reference.
        x, y = mini_batch.src, mini_batch.tgt[0][:, 1:] # [0] -> tensors (lengths not included)
                                                        # [:, 1:] -> groundtruth (without <BOS>)
        # |x| = (batch_size, max_length + 1)
        # |y| = (batch_size, length??)

        with autocast(not engine.config.off_autocast):
            # Take feed-forward
            # Similar as before, the input of decoder does not have EOS token.
            # Thus, remove EOS token for decoder input.
            y_hat = engine.model(x, mini_batch.tgt[0][:, :-1]) # input for Decoder (without <EOS>)
            # |y_hat| = (batch_size, length, output_size)

            loss = engine.crit(
                y_hat.contiguous().view(-1, y_hat.size(-1)),
                y.contiguous().view(-1)
            )
            # See get_crit function in train.py regarding the line below.
            backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update) # the first div -> to apply mean
                                                                                          # the second div -> due to grad accumulation

        if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
            engine.scaler.scale(backward_target).backward()
        else:
            backward_target.backward() # Note that GradScaler only works in GPU.

        word_count = int(mini_batch.tgt[1].sum())
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        if engine.state.iteration % engine.config.iteration_per_update == 0 and \
            engine.state.iteration > 0:
            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad_norm,
            )
            # Take a step of gradient descent.
            if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
                # Use scaler instead of engine.optimizer.step() if using GPU.
                engine.scaler.step(engine.optimizer)
                engine.scaler.update()
            else:
                engine.optimizer.step()

        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl,
            # Even though training process shows no problems, |param| can be shown as nan or inf.
            # That is, we cannot apply mean to nan or inf.
            # This is the reason why we need to take actions like below.
            '|param|': p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
            '|g_param|': g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device
            mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
            mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)

            with autocast(not engine.config.off_autocast):
                y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
                # |y_hat| = (batch_size, n_classes)
                loss = engine.crit(
                    y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1),
                )
        
        word_count = int(mini_batch.tgt[1].sum())
        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl,
        }

    @staticmethod
    def attach(
        train_engine, validation_engine,
        training_metric_names = ['loss', 'ppl', '|param|', '|g_param|'],
        validation_metric_names = ['loss', 'ppl'],
        verbose=VERBOSE_BATCH_WISE,
    ):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_loss = engine.state.metrics['loss']

                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} ppl={:.2f}'.format(
                    engine.state.epoch,
                    avg_p_norm,
                    avg_g_norm,
                    avg_loss,
                    np.exp(avg_loss),
                ))

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_loss = engine.state.metrics['loss']

                print('Validation - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                    avg_loss,
                    np.exp(avg_loss),
                    engine.best_loss,
                    np.exp(engine.best_loss),
                ))

    @staticmethod
    def resume_training(engine, resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch - 1)

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_loss = train_engine.state.metrics['loss']
        avg_valid_loss = engine.state.metrics['loss']

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model_fn.split('.')
        
        model_fn = model_fn[:-1] + ['%02d' % train_engine.state.epoch,
                                    '%.2f-%.2f' % (avg_train_loss,
                                                   np.exp(avg_train_loss)
                                                   ),
                                    '%.2f-%.2f' % (avg_valid_loss,
                                                   np.exp(avg_valid_loss)
                                                   )
                                    ] + [model_fn[-1]]

        model_fn = '.'.join(model_fn)

        # Unlike other tasks, we need to save current model, not best model.
        torch.save(
            {
                'model': engine.model.state_dict(),
                'opt': train_engine.optimizer.state_dict(),
                'config': config,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
            }, model_fn
        )
    

class SingleTrainer():

    def __init__(self, target_engine_class, config):
        self.target_engine_class = target_engine_class
        self.config = config

    def train(
        self,
        model, crit, optimizer,
        train_loader, valid_loader,
        src_vocab, tgt_vocab,
        n_epochs,
        lr_scheduler=None
    ):
        # Declare train and validation engine with necessary objects.
        train_engine = self.target_engine_class(
            self.target_engine_class.train,
            model,
            crit,
            optimizer,
            lr_scheduler,
            self.config
        )
        validation_engine = self.target_engine_class(
            self.target_engine_class.validate,
            model,
            crit,
            optimizer=None,
            lr_scheduler=None,
            config=self.config
        )

        # Do necessary attach procedure to train & validation engine.
        # Progress bar and metric would be attached.
        self.target_engine_class.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        # After every train epoch, run 1 validation epoch.
        # Also, apply LR scheduler if it is necessary.
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

            if engine.lr_scheduler is not None:
                engine.lr_scheduler.step()

        # Attach above call-back function.
        # That is, the validation process is nothing but a call-back in the middle of training.
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation, # run_validation(validation_engine, valid_loader)
            validation_engine,
            valid_loader
        )
        # Attach other call-back function for initiation of the training.
        train_engine.add_event_handler(
            Events.STARTED,
            self.target_engine_class.resume_training, # self.target_engine_class.resume_training(self.config.init_epoch)
            self.config.init_epoch,
        )

        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, self.target_engine_class.check_best
        )
        
        # Attach model save procedure for every end of validation epoch.
        # This is because the lowest loss does not guarantee the best model.
        # For more detailed reasons, learn the inference for NLG.
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            self.target_engine_class.save_model,
            train_engine,
            self.config,
            src_vocab,
            tgt_vocab,
        )

        # Start training.
        train_engine.run(train_loader, max_epochs=n_epochs)

        return model
