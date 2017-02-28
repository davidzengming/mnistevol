import tensorflow as tf

def train(mnist, accuracy, saver):
	sess.run(tf.global_variables_initializer())
	for i in range(20000):
	    batch = mnist.train.next_batch(50)
	    if i % 100 == 0:
	        train_accuracy = accuracy.eval(feed_dict = {x:batch[0], y_: batch[1], keep_prob: 1.0})
	        print("step %d, training accuracy %g" % (i, train_accuracy))
	    train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})

	# Test step
	print("test accuracy %g" % accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

	# Save network
	save_path = saver.save(sess, "save_2")
	print("Model saved as: %s" % save_path)
