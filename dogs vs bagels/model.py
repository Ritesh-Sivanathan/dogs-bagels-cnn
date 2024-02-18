from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf

from preprocess import preprocess_image, get_preprocessed

processor = get_preprocessed('C:\\Users\\User\\Documents\\Development\\AI\\Tinygrad\\dogs vs bagels\\dogs', 'C:\\Users\\User\\Documents\\Development\\AI\\Tinygrad\\dogs vs bagels\\bagels')
bagel_set, dog_set = processor[0]
all_datasets = bagel_set + dog_set
bagel_label, dog_label = processor[1]
all_labels = bagel_label + dog_label

dataset = tf.data.Dataset.from_tensor_slices((all_datasets, all_labels))

batch_size = 2
dataset = dataset.shuffle(len(all_datasets)).batch(batch_size)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(406, 612, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

dataset = dataset.shuffle(len(all_datasets))
validation_size = int(0.2 * len(all_datasets))
train_dataset = dataset.skip(validation_size)
validation_dataset = dataset.take(validation_size)

history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
test_loss, test_acc = model.evaluate(dataset)

print('Test accuracy:', test_acc)

model.save('C:\\Users\\User\\Documents\\Development\\AI\\Tinygrad\\dogs vs bagels\\model\\model_weights.h5')
