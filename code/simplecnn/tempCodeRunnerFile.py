optimizer = 'adam'
# loss = "categorical_crossentropy"

# model = models.build_model_01((224,224, 3), 4)

# model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=10
# )

# model.save("resnet_orientation.h5")

# model_score = train.evaluate_model(model, x_test_final, y_test_final)
# print('Test loss:', model_score[0])
# print('Test accuracy:', model_score[1])