import numpy as np
import matplotlib as plt
import tensorflow as tf
#from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)


print("-Training-set:\t\t{}".format(len(mnist.train.labels)))

print("-Test-set:\t\t{}".format(len(mnist.test.labels)))

print("-Validation-set:\t\t{}".format(len(mnist.validation.labels)))

testlabels=mnist.test.labels  #STORING THE TESTING VECTOR LABELS IN VARIABLE

#print(mnist.test.labels[0:5, :]) #One-Hot Encoding

mnist.test.cls=np.array([label.argmax() for label in testlabels])

print(mnist.test.cls[0:5])

img_size=28 #The Mnist images are 28*28 pixels in dimensions\

img_flat_size=img_size*img_size #they are stored in one-dimensional arrays in length

img_shape=(img_size,img_size) #tuple with height and width to resize the aarray\

num_class=10

def plot_images(images, cls_true, cls_pred):

    assert len(images)==len(cls_true)==9

    #Create figure with 3*3 sub-plots
    fig, axes= plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)

    for i, ax in enumerate(axes.flat) :
        ax.imshow(images[i].reshape(img_shape, cmap='binary'))

        if cls_pred is None:
            xlabel ="True: {0}".format(cls_true[i])
        else:
            xlabel="True: {0},Pred: {1}".format(cls_true[i],cls_pred)

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

        images=mnist.test.images[0:9]# first 10 images frim the dataset

        cls_true=mnist.test.cls[0:9] # get the true classes for those images
        
        

        print(plot_images(images=images,cls_true=cls_true))#plt the images and labels using the helper function

        x=tf.placeholder(tf.float32, [None,img_flat_size])#placeholder for input images

        y_true=tf.placeholder(tf.float32, [None,num_class])#placeholders for true labels of the input images

        y_true_cls=tf.placeholder((tf.int64,[None]))#placeholder for true class of the images

        weights=tf.Variable(tf.zeros([img_flat_size,num_class])) #giving weights to the input images

        bias=tf.Variable(tf.zeros([num_class]))

        logits=tf.matmul(x,weights)+ bias

        y_pred=tf.nn.softmax(logits)

        y_pred_cls = tf.argmax(y_pred,dimension=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)

        cost = tf.reduce_sum(cross_entropy)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

        correct_predication = tf.equal(y_pred_cls,y_true_cls)

        accuracy=tf.reduce_mean(tf.cast(correct_predication,tf.float32))


        Sess = tf.Session()
        Sess.run(tf.initialize_all_variables())

        batch_size=100

        def optimizer(num_iterations):

            for i in range(num_iterations):

                x_batch,y_true_batch = mnist.train.next_batch(batch_size)

                feed_dict_train = {x:x_batch,y_true:y_true_batch}
                Sess.run(optimizer,feed_dict=feed_dict_train)




        feed_dict_test = {x:mnist.test.images,y_true:mnist.test.labels,y_true_cls:mnist.test.cls}


        def acurracy():
            acc=Sess.run(accuracy,feed_dict=feed_dict_test)
            print("Accuracy on test set:{0:.1%}".format(acc))

        optimizer(num_iterations=1)
        accuracy()
        
