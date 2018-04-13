# Imports
import numpy as np
import tensorflow as tf
import dataset

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

tf.logging.set_verbosity(tf.logging.INFO)

#Prepare input data
# classes = ['dogs','cats']
class_list = [ "Brushing", "Cutting", "Jumping", "Lunges", "Wall" ]
num_classes = len(class_list)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 120
num_channels = 3
train_path='data/UCF-101/folds'


def eval_confusion_matrix(labels, predictions):
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=5)

        con_matrix_sum = tf.Variable(tf.zeros(shape=(5,5), dtype=tf.int32),
                                            trainable=False,
                                            name="confusion_matrix_result",
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])


        update_op = tf.assign_add(con_matrix_sum, con_matrix)

        return tf.convert_to_tensor(con_matrix_sum), update_op

def cnn_model_fn_old(features, labels, mode, params):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 120, 120, 3])

    
    #I don't think our first layer needs this big kernel size, probably too big for our input size. Maybe 5 is enough
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[5, 5],
        strides=(2,2),
        padding="valid",
        activation=tf.nn.relu)

    # norm1 = tf.nn.lrn(conv1, 5, 2, 0.0001, 0.75)

    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3,3], strides=2)


    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=256,
        kernel_size=[3, 3],
        #strides=(2,2),
        padding="same",
        activation=tf.nn.relu)

    # norm2 = tf.nn.lrn(conv2, 5, 2, 0.0001, 0.75)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3,3], strides=2)


    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)


    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)


    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3,3], strides=2)

    print(pool3) #//[batch, 6, 6, 512]

    pool3_flat = tf.reshape(pool3, [-1, 6 * 6 * 512])

    #Not sure if relu ois the correct activation function, probably in paper
    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096, activation=tf.nn.relu)
    
    #Random rate now, in paper is correct one
    dropout1 = tf.layers.dropout(inputs=dense1, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN)

    
    #Not sure if relu ois the correct activation function, probably in paper
    dense2 = tf.layers.dense(inputs=dropout1, units=2048, activation=tf.nn.relu)
    
    #Random rate now, in paper is correct one
    dropout2 = tf.layers.dropout(inputs=dense2, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout2, units=5)


    print(logits)

    predictions= {
        "classes":  tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")        
    }

    print(predictions)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5)
    # a = tf.Print(logits, [logits])
    # a.eval()
    # for i in range(0,4):
    #     for j in range(0,5):
    #         print(logits[i][j])
        # print(logits[i])
    # print(labels)
    # print(logits)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits
    )


    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #This changes over the iterations
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    P = tf.metrics.precision(
        labels=labels, predictions=predictions["classes"])
    R = tf.metrics.recall(
        labels=labels, predictions=predictions["classes"])

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
            
        ) ,
        "recall": R 
        ,
        "precision": P
        ,
        "confusion_matrix": eval_confusion_matrix(labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )

# About learning rate: The learning rate is initially set to 10e−2, and then decreased according to a fixed schedule, 
#  which is kept the same for all training sets. Namely, when training a ConvNet from scratch, the rate is changed to 10e−3 
#  after 50K iterations, then to 10e−4 after 70K iterations, and training is stopped after 80K iterations. 
#  In the fine-tuning scenario, the rate is changed to 10e−3 after 14K iterations, and training stopped after 20K iterations.

def cnn_model_fn_new(features, labels, mode, params):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 120, 120, 3])

    
    #I don't think our first layer needs this big kernel size, probably too big for our input size. Maybe 5 is enough
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[7, 7],
        strides=(2,2),
        padding="valid",
        activation=tf.nn.relu)

    # norm1 = tf.nn.lrn(conv1, 5, 2, 0.0001, 0.75)

    
    #pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3,3], strides=2)


    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=256,
        kernel_size=[5, 5],
        #strides=(2,2),
        padding="valid",
        activation=tf.nn.relu)

    # norm2 = tf.nn.lrn(conv2, 5, 2, 0.0001, 0.75)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3,3], strides=2)


    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)


    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)


    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3,3], strides=2)

    print(pool3) #//[batch, 6, 6, 512]

    pool3_flat = tf.reshape(pool3, [-1, 12 * 12 * 512])

    #Not sure if relu ois the correct activation function, probably in paper
    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096, activation=tf.nn.relu)
    
    #Random rate now, in paper is correct one
    dropout1 = tf.layers.dropout(inputs=dense1, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN)

    
    #Not sure if relu ois the correct activation function, probably in paper
    dense2 = tf.layers.dense(inputs=dropout1, units=2048, activation=tf.nn.relu)
    
    #Random rate now, in paper is correct one
    dropout2 = tf.layers.dropout(inputs=dense2, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout2, units=5)


    print(logits)

    predictions= {
        "classes":  tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")        
    }

    print(predictions)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5)
    # a = tf.Print(logits, [logits])
    # a.eval()
    # for i in range(0,4):
    #     for j in range(0,5):
    #         print(logits[i][j])
        # print(logits[i])
    # print(labels)
    # print(logits)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits
    )


    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #This changes over the iterations
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    P = tf.metrics.precision(
        labels=labels, predictions=predictions["classes"])
    R = tf.metrics.recall(
        labels=labels, predictions=predictions["classes"])

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
            
        ) ,
        "recall": R 
        ,
        "precision": P
        ,
        "confusion_matrix": eval_confusion_matrix(labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )

def cnn_model_fn_newnew(features, labels, mode, params):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 120, 120, 3])

    
    #I don't think our first layer needs this big kernel size, probably too big for our input size. Maybe 5 is enough
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[7, 7],
        strides=(2,2),
        padding="valid",
        activation=tf.nn.relu)

    # norm1 = tf.nn.lrn(conv1, 5, 2, 0.0001, 0.75)

    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3,3], strides=2)


    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=256,
        kernel_size=[5, 5],
        strides=(2,2),
        padding="valid",
        activation=tf.nn.relu)

    # norm2 = tf.nn.lrn(conv2, 5, 2, 0.0001, 0.75)

    #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3,3], strides=2)


    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)


    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)


    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3,3], strides=2)

    print(pool3) #//[batch, 6, 6, 512]

    pool3_flat = tf.reshape(pool3, [-1, 20 * 20 * 512])

    #Not sure if relu ois the correct activation function, probably in paper
    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096, activation=tf.nn.relu)
    
    #Random rate now, in paper is correct one
    dropout1 = tf.layers.dropout(inputs=dense1, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN)

    
    #Not sure if relu ois the correct activation function, probably in paper
    dense2 = tf.layers.dense(inputs=dropout1, units=2048, activation=tf.nn.relu)
    
    #Random rate now, in paper is correct one
    dropout2 = tf.layers.dropout(inputs=dense2, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout2, units=5)


    print(logits)

    predictions= {
        "classes":  tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")        
    }

    print(predictions)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5)
    # a = tf.Print(logits, [logits])
    # a.eval()
    # for i in range(0,4):
    #     for j in range(0,5):
    #         print(logits[i][j])
        # print(logits[i])
    # print(labels)
    # print(logits)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits
    )


    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #This changes over the iterations
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    P = tf.metrics.precision(
        labels=labels, predictions=predictions["classes"])
    R = tf.metrics.recall(
        labels=labels, predictions=predictions["classes"])

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
            
        ) ,
        "recall": R 
        ,
        "precision": P
        ,
        "confusion_matrix": eval_confusion_matrix(labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )

def run_model(model, dropout_rate, num_epochs, f):

    result_array = []
    
    for i in range(1,6):

        # We shall load all the training and validation images and labels into memory using openCV and use that during training
        #TODO we need to do cross validation
        data = dataset.read_train_sets(train_path, i, False)


        # print("Complete reading input data. Will Now print a snippet of it")
        # print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
        # print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

        classifier = None

        # Create the estimator        
        if model == 0:
            classifier = tf.estimator.Estimator(
                model_fn=cnn_model_fn_old,
                params={
                    'dropout_rate': dropout_rate
                }
            )
        elif model == 1:
            classifier = tf.estimator.Estimator(
                model_fn=cnn_model_fn_new,
                params={
                    'dropout_rate': dropout_rate
                }
            )
        else:
            classifier = tf.estimator.Estimator(
                model_fn=cnn_model_fn_newnew,
                params={
                    'dropout_rate': dropout_rate
                }
            )

        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50
        )

        # print("shape of images: ")
        # print(data.train.images.shape)


        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": data.train.images},
            y=data.train.labels,
            batch_size=16,
            num_epochs=num_epochs,
            shuffle=True
        )

        classifier.train(
            input_fn=train_input_fn,
            #steps=1000,
            hooks=[logging_hook]
        )

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": data.valid.images},
            y=data.valid.labels,
            num_epochs=1,
            shuffle=False
        )

        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        f.write(eval_results)

        result_array.append(eval_results)

    return result_array

def main(unused_argv):
    f = open("outputfile.txt", "w")

    # TODO
    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    for m in range(2,3):
        f.write("Starting model " + str(m) + " now")
        
        for dropout_rate in np.arange(0.4, 0.9, 0.25):

            f.write("Starting with dropout " + str(dropout_rate) + " now")     

            for num_epochs in range(5, 45, 10):

                f.write("Starting with num_epochs " + str(num_epochs) + " now")                

                result_array = run_model(m, dropout_rate, num_epochs, f)
                
                fscore = [0.0,0.0,0.0,0.0,0.0]
                avg_acc = 0.0
                avg_prec = 0.0
                avg_rec = 0.0
                avg_fscore = 0.0

                for i in range(0,5):
                    avg_acc += result_array[i].get("accuracy")
                    avg_prec += result_array[i].get("precision")
                    avg_rec += result_array[i].get("recall")
                    fscore[i] = 2 * result_array[i].get("recall") *result_array[i].get("precision")/(result_array[i].get("recall") + result_array[i].get("precision"))
                    avg_fscore += fscore[i]

                avg_acc /= 5
                avg_prec /= 5
                avg_rec /= 5
                avg_fscore /= 5

                f.write("avg_acc: " + str(avg_acc))
                f.write("avg_prec: " + str(avg_prec))
                f.write("avg_rec: " + str(avg_rec))
                f.write("avg_fscore: " + str(avg_fscore))


    f.close()

# Our application logic will be added here

if __name__ == "__main__":
  tf.app.run()
