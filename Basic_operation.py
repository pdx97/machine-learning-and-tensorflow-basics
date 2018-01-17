import tensorflow as tf

a=tf.constant(2)

b=tf.constant(3)

Sess=tf.Session()#to access the tensorflow backend we need to activate the session

c=tf.add(a,b)# adds the two constants

print(Sess.run(c))
