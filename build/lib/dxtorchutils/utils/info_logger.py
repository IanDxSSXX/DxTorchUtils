from torch.utils.tensorboard import SummaryWriter
import random


class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def log_training(self, loss, learning_rate, duration, iteration):
            self.add_scalar("Training/Loss", loss, iteration)
            self.add_scalar("Training/LearningRate", learning_rate, iteration)
            self.add_scalar("Training/Duration", duration, iteration)

    def log_metric(self, metric, metric_name, loss,  epoch):
        self.add_scalar("Training/Eval/{}".format(metric_name), metric, epoch)
        self.add_scalar("Training/Eval/Loss", loss, epoch)


    def log_validation_img(self, loss, prediction, target, iteration):
        self.add_scalar("Validation/Loss", loss, iteration)
        self.add_image("Validation/Prediction", prediction, iteration)
        self.add_image("Validation/Target", target, iteration)

    def log_validation_label(self, loss, prediction, target, iteration):
        self.add_scalar("Validation/Loss", loss, iteration)
        if prediction == target:
            self.add_scalar("Validation/Pred==Target", 1, iteration)
        else:
            self.add_scalar("Validation/Pred==Target", 0, iteration)
