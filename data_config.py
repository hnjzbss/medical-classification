import os

label_list = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
              "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
              "Pneumonia", "Pneumothorax", "Support Devices"]

mimic_csv_path = "./data_file"
mimic_data_path = "/data/mimic_cxr/images512"


class ClassificationConfig():
    epochs = 200
    num_classes = 14
    batch_size = 64
    data_class = "MIMIC"
    debug = False
    display_f1 = True
    label_list = label_list

    def __init__(self, data_class=data_class):
        self.data_class = data_class
        self.csv_path = mimic_csv_path
        self.data_path = mimic_data_path
        self.save_weight_name = os.path.join("./weights", self.data_class) + "_best_weight.pth"
        self.train_log_name = os.path.join("./train_logs", self.data_class) + ".txt"
        self.test_log_name = os.path.join("./test_logs", self.data_class) + ".txt"
        self.acc_curve_path = os.path.join("./result_figure", self.data_class) + "_acc.jpg"
        self.loss_curve_path = os.path.join("./result_figure", self.data_class) + "_loss.jpg"
        self.confusion_matrics_path = os.path.join("./result_figure", self.data_class) + "_confusion_matrics.jpg"

        self.mkdir_weight_path()

    def mkdir_weight_path(self):
        if not os.path.exists("./weights"):
            os.makedirs("./weights")
        if not os.path.exists("./train_logs"):
            os.makedirs("./train_logs")
        if not os.path.exists("./test_logs"):
            os.makedirs("./test_logs")
        if not os.path.exists("./result_figure"):
            os.makedirs("./result_figure")

    def save_train_log(self, train_log_name, args, logs_info):
        with open(train_log_name, "w") as f:
            f.write(str(args))
            f.write("\n")
            for info in logs_info:
                f.write(str(info) + ("\n"))

    def save_test_log(self, test_log_name, logs_info):
        with open(test_log_name, "w") as f:
            for info in logs_info:
                f.write(str(info) + "\n")
