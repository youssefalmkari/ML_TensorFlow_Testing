from __future__ import print_function
import tensorflow as tf

# Create TensorFlow constant nodes
print("Constant nodes:")
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # dtype is implicitly tf.float32
print(node1, node2)
print("\n")

# Create a TensorFlow Session to evaluate nodes
print("Session evaluation of constant nodes:")
sess = tf.Session()
print(sess.run([node1, node2]))
print("\n") 

# Combine nodes using Operations and evaluate in a session
print("Combined node3 from adding node1 & node2")
node3 = tf.add(node1, node2)
print("node3:", node3)
print("Session evaluation of node3:")
print("sess.run(node3):", sess.run(node3))
print("\n")

# Create two placeholders and add them together
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a,b)

# Evaluate graph with multiple inputs by feeding concrete values
# as arguments (feed_dict) to the Session run method
print("Session evaluation of adder_node using discrete arguments:")
print("args: 3, 4 --> ",sess.run(adder_node, {a: 3, b: 4.5}))
print("args: [1,3], [2,4] --> ",sess.run(adder_node, {a:[1,3], b:[2,4]}))
print("\n")

# Increase graph complexity by adding another Operation
print("Add another operation to adder_node:")
add_and_triple = adder_node * 3
print("add_and_triple w/ args: 3. 4.5 --> ", sess.run(add_and_triple, {a: 3, b: 4.5}))

# Variables allow us to add trainable parameters to a graph
# They are constructed with a type and intitial value
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

# Must initialize all variables you must call a special operation:
init = tf.global_variables_initializer()
sess.run(init)
print("linear_model: W*x + b --> ", sess.run(linear_model, {x: [1, 2, 3, 4]}))