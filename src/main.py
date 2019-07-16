import tensorflow as tf
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm

###############
#PREPARE DATA
###############
def prepare_data():
    X_train = []
    Y_train = []
    for filename in tqdm(glob.glob('../pokemon/*.png')):
        im = Image.open(filename).convert('F')
        im2 = Image.open(filename).convert('RGB')
        im = np.array(im)
        im2 = np.array(im2)
        X_train.append(im)
        Y_train.append(im2)

    X_train = np.array(X_train)
    X_train.reshape((819, 256, 256, 1))
    Y_train = np.array(Y_train)
    Y_train.reshape((819, 256, 256, 3))
    return X_train, Y_train


###############
#GENERATOR
###############
def create_generator(img_shape):
    noise_shape = (10, 256, 256, 1)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=(655360, 1)))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Dense(np.prod(img_shape), activation='tanh'))
    model.add(tf.keras.layers.Reshape((img_shape)))

    model.summary()

    noise = tf.keras.layers.Input(shape=noise_shape)
    img = model(noise)

    return tf.keras.Model(noise, img)


###############
#DISCRIMINATOR
###############
def create_discriminator(img_shape):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=(1966080, 1)))
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    input = tf.keras.layers.Input(shape=img_shape)
    img = model(input)

    return tf.keras.Model(input, img)


###############
#SAVE IMAGES
###############
def save_imgs(generator, epoch, X_train):
    r, c = 1, 1
    idx = np.random.randint(0, X_train.shape[0], 1)
    imgs = X_train[idx]
    gen_imgs = generator.predict(imgs)

    gen_imgs = 0.5 * gen_imgs + 0.5
    gen_imgs = 255 * gen_imgs

    gen_imgs = gen_imgs.reshape((256 ,256))

    new_im = Image.fromarray(gen_imgs)
    new_im = new_im.convert("RGB")
    new_im.save("../outputPokemon/"+str(epoch)+".jpg")


###############
#TRAIN
###############
def train(epochs, batch_size, generator, discriminator, combined, save_interval=50):
    X_train, Y_train = prepare_data()

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train.reshape((819, 256, 256, 1))

    Y_train = (Y_train.astype(np.float32) - 127.5) / 127.5
    Y_train.reshape((819, 256, 256, 3))

    half_batch_size = int(batch_size / 2)

    for epoch in tqdm(range(epochs)):
        #Train discriminator
        idx = np.random.randint(0, Y_train.shape[0], half_batch_size)
        imgs = Y_train[idx]

        idx2 = np.random.randint(0, X_train.shape[0], half_batch_size)
        imgs2 = X_train[idx]
        gen_imgs = generator.predict(imgs2)

        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


        #Train generator
        idx2 = np.random.randint(0, X_train.shape[0], half_batch_size)
        imgs2 = X_train[idx]
        valid_y = np.array([1] * batch_size)

        g_loss = combined.train_on_batch(imgs2, valid_y)

        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        if epoch % save_interval == 0:
            save_imgs(generator, epoch)


generator = create_generator((10, 256, 256, 3))
generator.compile(loss='binary_crossentropy', optimizer='adam')

discriminator = create_discriminator((10, 256, 256, 3))
discriminator.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
discriminator.trainable = False

z = tf.keras.layers.Input(shape=(10, 256, 256, 1))
img = generator(z)
valid = discriminator(img)

combined = tf.keras.Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer='adam')


train(2000, 20, generator, discriminator, combined)
