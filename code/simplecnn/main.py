import models as models

"""
Here you can choose which model to train from the list
"""

print("Models list: \nmodel01 \nmodel02 \n")
model = input("Please insert to model you would like to train: ")

if model == "model01":
    model_01 = models.build_model_01((426, 320, 3), 4)
elif model == "model02":
    model_02 = models.build_model_02((426, 320, 3), 4)