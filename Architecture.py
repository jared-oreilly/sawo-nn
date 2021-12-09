from tensorflow.keras.utils import plot_model
import tensorflow as tf

class Architecture:
    def __init__(self, d, weights, sizes):
        layers = []

        x = tf.keras.layers.Input(shape=(sizes["i"],), name="i_0")
        layers.append([x])

        num_input_layers = 1
        all_input_layers = [x]

        for i in range(1, len(d)):

            before_symbols = d[i-1]
            cur_symbols = d[i]

            if len(before_symbols) == 1 and len(cur_symbols) == 1:
                x_symbol = before_symbols[0]
                x = layers[i-1][0]

                y_symbol = cur_symbols[0]
                if y_symbol == "i":
                    y = tf.keras.layers.Input(shape=(sizes["i"],), name=("i_"+str(num_input_layers)))
                    num_input_layers += 1
                    all_input_layers.append(y)
                else:
                    y_dash = tf.keras.layers.Dense(sizes[y_symbol], name=y_symbol)
                    y = y_dash(x)
                    y_dash.set_weights(weights[x_symbol+""+y_symbol])

                layers.append([y])
            
            elif len(before_symbols) == 1 and len(cur_symbols) > 1:
                x_symbol = before_symbols[0]
                x = layers[i-1][0]

                y_arr = []
                for k in range(len(cur_symbols)):
                    y_k_symbol = cur_symbols[k]

                    if y_k_symbol == "i":
                        y = tf.keras.layers.Input(shape=(sizes["i"],), name=("i_"+str(num_input_layers)))
                        num_input_layers += 1
                        all_input_layers.append(y)
                    else:
                        y_dash = tf.keras.layers.Dense(sizes[y_k_symbol], name=y_k_symbol)
                        y = y_dash(x)
                        y_dash.set_weights(weights[x_symbol+""+y_k_symbol])

                    y_arr.append(y)

                layers.append(y_arr)

            elif len(before_symbols) > 1 and len(cur_symbols) >= 1:

                z_arr = []
                for j in range(len(before_symbols)):
                    x_j_symbol = before_symbols[j]
                    x_j = layers[i-1][j]

                    z_j_arr = []
                    for k in range(len(cur_symbols)):
                        y_k_symbol = cur_symbols[k]

                        if y_k_symbol != "i":
                            z_j_k_dash = tf.keras.layers.Dense(sizes[y_k_symbol], name=(x_j_symbol+"2"+y_k_symbol))
                            z_j_k = z_j_k_dash(x_j)
                            z_j_k_dash.set_weights(weights[x_j_symbol+""+y_k_symbol])

                            z_j_arr.append(z_j_k)
                        else:
                            z_j_arr.append(None)

                    z_arr.append(z_j_arr)

                y_arr = []
                for k in range(len(cur_symbols)):
                    y_k_symbol = cur_symbols[k]

                    if y_k_symbol == "i":
                        y_k = tf.keras.layers.Input(shape=(sizes["i"],), name=("i_"+str(num_input_layers)))
                        num_input_layers += 1
                        all_input_layers.append(y_k)
                    else:
                        z_arr_for_k = [z_j_arr[k] for z_j_arr in z_arr]
                        y_k = tf.keras.layers.Average(name=y_k_symbol)(z_arr_for_k)

                    
                    y_arr.append(y_k)

                layers.append(y_arr)

        self.num_input_layers = len(all_input_layers)
        self.model = tf.keras.Model(inputs=all_input_layers, outputs=layers[len(layers)-1][0], name=("_".join(d)))
        
        #self.model.summary()
        #plot_model(self.model, 'model.png', show_shapes=True)
        
    
    def print(self):
        self.model.summary()
    
    def error(self, type, X, y):

        duplicated_X = [X for _ in range(self.num_input_layers)]
        y_ = self.model(duplicated_X)

        err_object = None
        if type == "mse":
            err_object = tf.keras.losses.MeanSquaredError()
            error = err_object(y_true=y, y_pred=y_)
        elif type == "cel":
            err_object = tf.keras.losses.CategoricalCrossentropy()
            error = err_object(y_true=y, y_pred=y_)
        elif type == "acc":
            err_object = tf.keras.metrics.CategoricalAccuracy()
            err_object.update_state(y, y_)
            error = err_object.result()
            

        return error

